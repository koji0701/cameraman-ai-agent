/* eslint global-require: off, no-console: off, promise/always-return: off */

/**
 * This module executes inside of electron's main process. You can start
 * electron renderer process from here and communicate with the other processes
 * through IPC.
 *
 * When running `npm run build` or `npm run build:main`, this file is compiled to
 * `./src/main.js` using webpack. This gives us some performance wins.
 */
import path from 'path';
import { app, BrowserWindow, shell, ipcMain, dialog } from 'electron';
import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import { spawn, ChildProcess } from 'child_process';
import fs from 'fs';
import MenuBuilder from './menu';
import { resolveHtmlPath } from './util';

class AppUpdater {
  constructor() {
    log.transports.file.level = 'info';
    autoUpdater.logger = log;
    autoUpdater.checkForUpdatesAndNotify();
  }
}

// Global state for video processing
let pythonProcess: ChildProcess | null = null;
let processingStatus = {
  isProcessing: false,
  progress: 0,
  stage: '',
  details: '',
};

let mainWindow: BrowserWindow | null = null;

// API Key management functions
const getConfigPath = () => {
  const userDataPath = app.getPath('userData');
  return path.join(userDataPath, 'config.json');
};

const loadConfig = () => {
  try {
    const configPath = getConfigPath();
    if (fs.existsSync(configPath)) {
      const configData = fs.readFileSync(configPath, 'utf8');
      return JSON.parse(configData);
    }
  } catch (error) {
    console.error('Error loading config:', error);
  }
  return {};
};

const saveConfig = (config: any) => {
  try {
    const configPath = getConfigPath();
    const configDir = path.dirname(configPath);
    if (!fs.existsSync(configDir)) {
      fs.mkdirSync(configDir, { recursive: true });
    }
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    return true;
  } catch (error) {
    console.error('Error saving config:', error);
    return false;
  }
};

// Original IPC example handler
ipcMain.on('ipc-example', async (event, arg) => {
  const msgTemplate = (pingPong: string) => `IPC test: ${pingPong}`;
  console.log(msgTemplate(arg));
  event.reply('ipc-example', msgTemplate('pong'));
});

// File selection handler
ipcMain.handle('select-video-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    title: 'Select Video File',
    filters: [
      {
        name: 'Video Files',
        extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'm4v'],
      },
      {
        name: 'All Files',
        extensions: ['*'],
      },
    ],
    properties: ['openFile'],
  });

  if (result.canceled) {
    return null;
  }

  return result.filePaths[0];
});

// Video processing handler
ipcMain.handle(
  'process-video',
  async (event, inputPath: string, outputPath: string, options: any) => {
    if (processingStatus.isProcessing) {
      throw new Error('A video is already being processed');
    }

    try {
      // Reset processing status
      processingStatus = {
        isProcessing: true,
        progress: 0,
        stage: 'Initializing',
        details: 'Starting Python backend...',
      };

      // Get the path to the Python backend launcher
      const appPath = app.getAppPath();
      const pythonLauncherPath = app.isPackaged
        ? path.join(
            process.resourcesPath,
            '..',
            '..',
            'app',
            'desktop_launcher.py',
          )
        : path.join(appPath, '..', 'app', 'desktop_launcher.py');

      console.log('Python launcher path:', pythonLauncherPath);
      console.log('App path:', appPath);
      console.log('Is packaged:', app.isPackaged);

      // Check if the launcher exists
      if (!fs.existsSync(pythonLauncherPath)) {
        throw new Error(`Python launcher not found at: ${pythonLauncherPath}`);
      }

      // Build command arguments for OpenCV-only processing
      const args = [
        pythonLauncherPath,
        'cli',
        inputPath,
        outputPath,
        '--quality',
        options.quality || 'medium',
        '--disable-streaming', // Always use OpenCV processing
      ];

      // Determine the correct Python interpreter path
      // First try to use the virtual environment's Python if available
      const pythonExePath = path.join(path.dirname(pythonLauncherPath), '..', '.venv', 'bin', 'python3');
      const finalPythonPath = fs.existsSync(pythonExePath) ? pythonExePath : 'python3';
      
      console.log('Executing command:', finalPythonPath, args);

      // Get the stored API key and set environment variable
      const config = loadConfig();
      const { geminiApiKey } = config;

      if (!geminiApiKey) {
        throw new Error(
          'Gemini API key not configured. Please set your API key in the settings.',
        );
      }

      const env = { ...process.env };
      env.GOOGLE_API_KEY = geminiApiKey;

      // Spawn the Python process
      pythonProcess = spawn(finalPythonPath, args, {
        cwd: path.dirname(pythonLauncherPath),
        stdio: ['pipe', 'pipe', 'pipe'],
        env,
      });

      // Handle process output
      pythonProcess.stdout?.on('data', (data) => {
        const output = data.toString();
        console.log('Python stdout:', output);

        // Parse progress information if available
        // Look for progress indicators in the output
        const progressMatch = output.match(/(\d+\.?\d*)%/);
        if (progressMatch) {
          processingStatus.progress = parseFloat(progressMatch[1]);
        }

        // Update stage based on output
        if (output.includes('Starting')) {
          processingStatus.stage = 'Starting';
        } else if (output.includes('Analyzing')) {
          processingStatus.stage = 'Analyzing';
        } else if (output.includes('Processing')) {
          processingStatus.stage = 'Processing';
        } else if (output.includes('Rendering')) {
          processingStatus.stage = 'Rendering';
        } else if (output.includes('Finalizing')) {
          processingStatus.stage = 'Finalizing';
        }

        processingStatus.details = output.trim().split('\n').pop() || '';

        // Send progress update to renderer
        mainWindow?.webContents.send('processing-progress', processingStatus);
      });

      pythonProcess.stderr?.on('data', (data) => {
        const error = data.toString();
        console.error('Python stderr:', error);
        processingStatus.details = `Error: ${error.trim()}`;
        mainWindow?.webContents.send('processing-progress', processingStatus);
      });

      // Handle process completion
      return new Promise((resolve, reject) => {
        pythonProcess!.on('close', (code) => {
          console.log(`Python process exited with code ${code}`);

          processingStatus.isProcessing = false;

          if (code === 0) {
            processingStatus.progress = 100;
            processingStatus.stage = 'Completed';
            processingStatus.details =
              'Video processing completed successfully!';
            mainWindow?.webContents.send(
              'processing-progress',
              processingStatus,
            );
            resolve({ success: true, outputPath });
          } else {
            processingStatus.stage = 'Failed';
            processingStatus.details = `Process failed with exit code ${code}`;
            mainWindow?.webContents.send(
              'processing-progress',
              processingStatus,
            );
            reject(new Error(`Python process failed with exit code ${code}`));
          }

          pythonProcess = null;
        });

        pythonProcess!.on('error', (err) => {
          console.error('Python process error:', err);
          processingStatus.isProcessing = false;
          processingStatus.stage = 'Failed';
          processingStatus.details = `Process error: ${err.message}`;
          mainWindow?.webContents.send('processing-progress', processingStatus);
          reject(err);
          pythonProcess = null;
        });
      });
    } catch (error) {
      processingStatus.isProcessing = false;
      processingStatus.stage = 'Failed';
      processingStatus.details = `Error: ${error}`;
      console.error('Process video error:', error);
      throw error;
    }
  },
);

// Processing status handler
ipcMain.handle('get-processing-status', async () => {
  return processingStatus;
});

// Cancel processing handler
ipcMain.handle('cancel-processing', async () => {
  if (pythonProcess) {
    pythonProcess.kill('SIGTERM');
    pythonProcess = null;
    processingStatus.isProcessing = false;
    processingStatus.stage = 'Cancelled';
    processingStatus.details = 'Processing cancelled by user';
    return true;
  }
  return false;
});

// Open output folder handler
ipcMain.handle('open-output-folder', async (event, filePath: string) => {
  shell.showItemInFolder(filePath);
});

// Utility handlers
ipcMain.handle('get-app-version', async () => {
  return app.getVersion();
});

ipcMain.handle(
  'show-error-dialog',
  async (event, title: string, content: string) => {
    return dialog.showErrorBox(title, content);
  },
);

ipcMain.handle(
  'show-info-dialog',
  async (event, title: string, content: string) => {
    return dialog.showMessageBox(mainWindow!, {
      type: 'info',
      title,
      message: content,
      buttons: ['OK'],
    });
  },
);

// API Key management handlers
ipcMain.handle('get-api-key', async () => {
  const config = loadConfig();
  return config.geminiApiKey || null;
});

ipcMain.handle('save-api-key', async (event, apiKey: string) => {
  const config = loadConfig();
  config.geminiApiKey = apiKey;
  return saveConfig(config);
});

ipcMain.handle('delete-api-key', async () => {
  const config = loadConfig();
  delete config.geminiApiKey;
  return saveConfig(config);
});

if (process.env.NODE_ENV === 'production') {
  const sourceMapSupport = require('source-map-support');
  sourceMapSupport.install();
}

const isDebug =
  process.env.NODE_ENV === 'development' || process.env.DEBUG_PROD === 'true';

if (isDebug) {
  require('electron-debug').default();
}

const installExtensions = async () => {
  const installer = require('electron-devtools-installer');
  const forceDownload = !!process.env.UPGRADE_EXTENSIONS;
  const extensions = ['REACT_DEVELOPER_TOOLS'];

  return installer
    .default(
      extensions.map((name) => installer[name]),
      forceDownload,
    )
    .catch(console.log);
};

const createWindow = async () => {
  if (isDebug) {
    await installExtensions();
  }

  const RESOURCES_PATH = app.isPackaged
    ? path.join(process.resourcesPath, 'assets')
    : path.join(__dirname, '../../assets');

  const getAssetPath = (...paths: string[]): string => {
    return path.join(RESOURCES_PATH, ...paths);
  };

  mainWindow = new BrowserWindow({
    show: false,
    width: 1024,
    height: 728,
    icon: getAssetPath('icon.png'),
    webPreferences: {
      preload: app.isPackaged
        ? path.join(__dirname, 'preload.js')
        : path.join(__dirname, '../../.erb/dll/preload.js'),
    },
  });

  mainWindow.loadURL(resolveHtmlPath('index.html'));

  mainWindow.on('ready-to-show', () => {
    if (!mainWindow) {
      throw new Error('"mainWindow" is not defined');
    }
    if (process.env.START_MINIMIZED) {
      mainWindow.minimize();
    } else {
      mainWindow.show();
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  const menuBuilder = new MenuBuilder(mainWindow);
  menuBuilder.buildMenu();

  // Open urls in the user's browser
  mainWindow.webContents.setWindowOpenHandler((edata) => {
    shell.openExternal(edata.url);
    return { action: 'deny' };
  });

  // Remove this if your app does not use auto updates
  // eslint-disable-next-line
  new AppUpdater();
};

/**
 * Add event listeners...
 */

app.on('window-all-closed', () => {
  // Respect the OSX convention of having the application in memory even
  // after all windows have been closed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app
  .whenReady()
  .then(() => {
    createWindow();
    app.on('activate', () => {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (mainWindow === null) createWindow();
    });
  })
  .catch(console.log);

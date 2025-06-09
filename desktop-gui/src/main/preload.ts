// Disable no-unused-vars, broken for spread args
/* eslint no-unused-vars: off */
import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';

export type Channels = 
  | 'ipc-example'
  | 'select-video-file'
  | 'process-video'
  | 'get-processing-status'
  | 'cancel-processing'
  | 'open-output-folder'
  | 'get-app-version'
  | 'show-error-dialog'
  | 'show-info-dialog';

const electronHandler = {
  ipcRenderer: {
    sendMessage(channel: Channels, ...args: unknown[]) {
      ipcRenderer.send(channel, ...args);
    },
    on(channel: Channels, func: (...args: unknown[]) => void) {
      const subscription = (_event: IpcRendererEvent, ...args: unknown[]) =>
        func(...args);
      ipcRenderer.on(channel, subscription);

      return () => {
        ipcRenderer.removeListener(channel, subscription);
      };
    },
    once(channel: Channels, func: (...args: unknown[]) => void) {
      ipcRenderer.once(channel, (_event, ...args) => func(...args));
    },
    // Convenience methods for common operations
    async invoke(channel: Channels, ...args: unknown[]): Promise<any> {
      return ipcRenderer.invoke(channel, ...args);
    },
  },
  // File operations
  selectVideoFile: () => ipcRenderer.invoke('select-video-file'),
  openOutputFolder: (path: string) => ipcRenderer.invoke('open-output-folder', path),
  
  // Video processing operations
  processVideo: (inputPath: string, outputPath: string, options: any) => 
    ipcRenderer.invoke('process-video', inputPath, outputPath, options),
  getProcessingStatus: () => ipcRenderer.invoke('get-processing-status'),
  cancelProcessing: () => ipcRenderer.invoke('cancel-processing'),
  
  // Utility functions
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  showErrorDialog: (title: string, content: string) => 
    ipcRenderer.invoke('show-error-dialog', title, content),
  showInfoDialog: (title: string, content: string) => 
    ipcRenderer.invoke('show-info-dialog', title, content),
};

contextBridge.exposeInMainWorld('electron', electronHandler);

export type ElectronHandler = typeof electronHandler;

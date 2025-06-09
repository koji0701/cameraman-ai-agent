import { ElectronHandler } from '../main/preload';

declare global {
  // eslint-disable-next-line no-var
  var electron: ElectronHandler;
}

export {};

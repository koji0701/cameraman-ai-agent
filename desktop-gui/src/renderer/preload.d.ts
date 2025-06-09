import { ElectronHandler } from '../main/preload';

declare global {
  // eslint-disable-next-line no-var, vars-on-top
  var electron: ElectronHandler;
}

export {};

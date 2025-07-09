import { configureStore } from '@reduxjs/toolkit'
import imageEditorReducer from './features/image-editor/imageEditorSlice'
import controlPanelReducer from './features/control-panel/controlPanelSlice'
import previewReducer from './features/preview/previewSlice'
import promptPanelReducer from './features/prompt-panel/promptPanelSlice'

export const makeStore = () => {
  return configureStore({
    reducer: {
      imageEditor: imageEditorReducer,
      controlPanel: controlPanelReducer,
      preview: previewReducer,
      promptPanel: promptPanelReducer,
    },
  })
}

// Infer the type of makeStore
export type AppStore = ReturnType<typeof makeStore>
// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<AppStore['getState']>
export type AppDispatch = AppStore['dispatch']

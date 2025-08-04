import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface ControlPanelState {
  generationIterations: number
  generationGuidanceScale: number
  img2imgStrength: number
  img2imgGuidanceScale: number
  img2imgInferenceSteps: number
  img2imgUseRefiner: boolean
  img2imgRefinerRatio: number
  img2imgRefinerInferenceSteps: number
  img2imgRefinerGuidanceScale: number
  inpaintStrength: number
  inpaintInferenceSteps: number
  inpaintGuidanceScale: number
  inpaintUseRefiner: boolean
  inpaintRefinerRatio: number
  inpaintRefinerInferenceSteps: number
  inpaintRefinerGuidanceScale: number
  maskCount: number
  maskIndex: number
  masked: boolean
  maskVisible: boolean
  alpha: number
  disabled: boolean
  strict: boolean
  reverseMask: boolean
  progress: number
}

const initialState: ControlPanelState = {
  generationIterations: 15,
  generationGuidanceScale: 7.5,
  img2imgStrength: 0.2,
  img2imgGuidanceScale: 7.5,
  img2imgInferenceSteps: 50,
  img2imgUseRefiner: false,
  img2imgRefinerRatio: 0.7,
  img2imgRefinerInferenceSteps: 30,
  img2imgRefinerGuidanceScale: 3.0,
  inpaintStrength: 0.2,
  inpaintInferenceSteps: 50,
  inpaintGuidanceScale: 7.5,
  inpaintUseRefiner: false,
  inpaintRefinerRatio: 0.85,
  inpaintRefinerInferenceSteps: 30,
  inpaintRefinerGuidanceScale: 7.5,
  maskCount: 0,
  maskIndex: 0,
  masked: false,
  maskVisible: true,
  alpha: 255,
  disabled: false,
  strict: false,
  reverseMask: false,
  progress: 0,
}

export const controlPanelSlice = createSlice({
  name: 'controlPanel',
  initialState,
  reducers: {
    setGenerationIterations: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.generationIterations = action.payload
    },
    setGenerationGuidanceScale: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.generationGuidanceScale = action.payload
    },
    setImg2imgStrength: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.img2imgStrength = action.payload
    },
    setImg2imgInferenceSteps: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.img2imgInferenceSteps = action.payload
    },
    setImg2imgGuidanceScale: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.img2imgGuidanceScale = action.payload
    },
    setImg2imgUseRefiner: (state: ControlPanelState, action: PayloadAction<boolean>) => {
      state.img2imgUseRefiner = action.payload
    },
    setImg2imgRefinerRatio: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.img2imgRefinerRatio = action.payload
    },
    setImg2imgRefinerInferenceSteps: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.img2imgRefinerInferenceSteps = action.payload
    },
    setImg2imgRefinerGuidanceScale: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.img2imgRefinerGuidanceScale = action.payload
    },
    setInpaintStrength: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintStrength = action.payload
    },
    setInpaintInferenceSteps: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintInferenceSteps = action.payload
    },
    setInpaintGuidanceScale: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintGuidanceScale = action.payload
    },
    setInpaintUseRefiner: (state: ControlPanelState, action: PayloadAction<boolean>) => {
      state.inpaintUseRefiner = action.payload
    },
    setInpaintRefinerRatio: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintRefinerRatio = action.payload
    },
    setInpaintRefinerInferenceSteps: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintRefinerInferenceSteps = action.payload
    },
    setInpaintRefinerGuidanceScale: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintRefinerGuidanceScale = action.payload
    },
    setMaskCount: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.maskCount = action.payload
    },
    setMaskIndex: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.maskIndex = action.payload
    },
    setMasked: (state: ControlPanelState, action: PayloadAction<boolean>) => {
      state.masked = action.payload
    },
    setAlpha: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.alpha = action.payload
    },
    toggleMaskVisible: (state: ControlPanelState, action: PayloadAction<boolean>) => {
      state.maskVisible = !state.maskVisible
    },
    toggleDisabled: (state: ControlPanelState, action) => {
      state.disabled = action.payload
    },

    toggleStrict: (state: ControlPanelState, action: PayloadAction<boolean>) => {
      state.strict = action.payload
    },
    toggleReverseMask: (state: ControlPanelState, action: PayloadAction<boolean>) => {
      state.reverseMask = action.payload
    },
  },
})

export const {
  setGenerationIterations,
  setGenerationGuidanceScale,
  setImg2imgStrength,
  setImg2imgInferenceSteps,
  setImg2imgGuidanceScale,
  setImg2imgUseRefiner,
  setImg2imgRefinerRatio,
  setImg2imgRefinerInferenceSteps,
  setImg2imgRefinerGuidanceScale,
  setInpaintStrength,
  setInpaintInferenceSteps,
  setInpaintUseRefiner,
  setInpaintRefinerRatio,
  setInpaintGuidanceScale,
  setInpaintRefinerInferenceSteps,
  setInpaintRefinerGuidanceScale,
  setMaskCount,
  setMaskIndex,
  setMasked,
  setAlpha,
  toggleMaskVisible,
  toggleDisabled,
  toggleStrict,
  toggleReverseMask,
} = controlPanelSlice.actions

export default controlPanelSlice.reducer

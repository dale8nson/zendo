import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface ControlPanelState {
  generationIterations: number
  generationGuidanceScale: number
  img2imgStrength: number
  img2imgGuidanceScale: number
  inpaintStrength: number
  inpaintGuidanceScale: number
  inpaintNoise: number
  inpaintNoiseOffset: number
  inpaintBlur: number
  maskCount: number
  maskIndex: number
  masked: boolean
  maskVisible: boolean
  alpha: number
  disabled: boolean
  strict: boolean
  reverseMask: boolean
  progress: number
  useNoise: boolean
}

const initialState: ControlPanelState = {
  generationIterations: 0,
  generationGuidanceScale: 7.5,
  img2imgStrength: 0.5,
  img2imgGuidanceScale: 7.5,
  inpaintStrength: 0.5,
  inpaintGuidanceScale: 7.5,
  inpaintNoise: 0.0,
  inpaintNoiseOffset: 0.5,
  inpaintBlur: 1,
  maskCount: 0,
  maskIndex: 0,
  masked: false,
  maskVisible: true,
  alpha: 255,
  disabled: false,
  strict: false,
  reverseMask: false,
  progress: 0,
  useNoise: false,
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
    setImg2imgGuidanceScale: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.img2imgGuidanceScale = action.payload
    },
    setInpaintStrength: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintStrength = action.payload
    },
    setInpaintGuidanceScale: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintGuidanceScale = action.payload
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
    setInpaintNoise: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintNoise = action.payload
    },
    setInpaintNoiseOffset: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintNoiseOffset = action.payload
    },
    setInpaintBlur: (state: ControlPanelState, action: PayloadAction<number>) => {
      state.inpaintBlur = action.payload
    },
    toggleStrict: (state: ControlPanelState, action: PayloadAction<boolean>) => {
      state.strict = action.payload
    },
    toggleReverseMask: (state: ControlPanelState, action: PayloadAction<boolean>) => {
      state.reverseMask = action.payload
    },
    toggleUseNoise: (state: ControlPanelState, action: PayloadAction<boolean>) => {
      state.useNoise = action.payload
    },
  },
})

export const {
  setGenerationIterations,
  setGenerationGuidanceScale,
  setImg2imgStrength,
  setImg2imgGuidanceScale,
  setInpaintStrength,
  setInpaintGuidanceScale,
  setInpaintNoise,
  setInpaintNoiseOffset,
  setInpaintBlur,
  setMaskCount,
  setMaskIndex,
  setMasked,
  setAlpha,
  toggleMaskVisible,
  toggleDisabled,
  toggleStrict,
  toggleReverseMask,
  toggleUseNoise,
} = controlPanelSlice.actions

export default controlPanelSlice.reducer

import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface PromptPanelState {
  generate_prompt: string
  generate_prompt2: string
  generate_negativePrompt: string
  generate_negativePrompt2: string
  img2img_prompt: string
  img2img_prompt2: string
  img2img_negativePrompt: string
  img2img_negativePrompt2: string
  inpaint_prompt: string
  inpaint_prompt2: string
  inpaint_negativePrompt: string
  inpaint_negativePrompt2: string
  default_prompt_tab: string
  selected_tab: string
}

const initialState: PromptPanelState = {
  generate_prompt: '',
  generate_prompt2: '',
  generate_negativePrompt: '',
  generate_negativePrompt2: '',
  img2img_prompt: '',
  img2img_prompt2: '',
  img2img_negativePrompt: '',
  img2img_negativePrompt2: '',
  inpaint_prompt: '',
  inpaint_prompt2: '',
  inpaint_negativePrompt: '',
  inpaint_negativePrompt2: '',
  default_prompt_tab: 'generate',
  selected_tab: 'generate',
}

export const PromptPanelSlice = createSlice({
  name: 'PromptPanel',
  initialState,
  reducers: {
    setGeneratePrompt: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.generate_prompt = action.payload
    },
    setGeneratePrompt2: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.generate_prompt2 = action.payload
    },
    setGenerateNegativePrompt: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.generate_negativePrompt = action.payload
    },
    setGenerateNegativePrompt2: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.generate_negativePrompt2 = action.payload
    },
    setImg2imgPrompt: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.img2img_prompt = action.payload
    },
    setImg2imgPrompt2: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.img2img_prompt2 = action.payload
    },
    setImg2imgNegativePrompt: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.img2img_negativePrompt = action.payload
    },
    setImg2imgNegativePrompt2: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.img2img_negativePrompt2 = action.payload
    },
    setInpaintPrompt: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.inpaint_prompt = action.payload
    },
    setInpaintPrompt2: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.inpaint_prompt2 = action.payload
    },
    setInpaintNegativePrompt: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.inpaint_negativePrompt = action.payload
    },
    setInpaintNegativePrompt2: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.inpaint_negativePrompt2 = action.payload
    },
    setDefaultPromptTab: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.default_prompt_tab = action.payload
    },
    setSelectedTab(state, action) {
      state.selected_tab = action.payload
    },
  },
})

export const {
  setGeneratePrompt,
  setGeneratePrompt2,
  setGenerateNegativePrompt,
  setGenerateNegativePrompt2,
  setImg2imgPrompt,
  setImg2imgPrompt2,
  setImg2imgNegativePrompt,
  setImg2imgNegativePrompt2,
  setInpaintPrompt,
  setInpaintPrompt2,
  setInpaintNegativePrompt,
  setInpaintNegativePrompt2,
  setDefaultPromptTab,
  setSelectedTab,
} = PromptPanelSlice.actions

export default PromptPanelSlice.reducer

import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface PromptPanelState {
  prompt: string
  prompt2: string
  negativePrompt: string
  negativePrompt2: string
}

const initialState: PromptPanelState = {
  prompt: '',
  prompt2: '',
  negativePrompt: '',
  negativePrompt2: '',
}

export const PromptPanelSlice = createSlice({
  name: 'PromptPanel',
  initialState,
  reducers: {
    setPrompt: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.prompt = action.payload
    },
    setPrompt2: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.prompt2 = action.payload
    },
    setNegativePrompt: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.negativePrompt = action.payload
    },
    setNegativePrompt2: (state: PromptPanelState, action: PayloadAction<string>) => {
      state.negativePrompt2 = action.payload
    },
  },
})

export const { setPrompt, setPrompt2, setNegativePrompt, setNegativePrompt2 } =
  PromptPanelSlice.actions

export default PromptPanelSlice.reducer

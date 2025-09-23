import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface EditorPanelState {
  captionScore: number | null
  captionScoreLoading: boolean
}

const initialState: EditorPanelState = {
  captionScore: null,
  captionScoreLoading: false,
}

const editorPanelSlice = createSlice({
  name: 'editorPanel',
  initialState,
  reducers: {
    setCaptionScore: (state, action: PayloadAction<number | null>) => {
      state.captionScore = action.payload
      state.captionScoreLoading = false
    },
    setCaptionScoreLoading: (state, action: PayloadAction<boolean>) => {
      state.captionScoreLoading = action.payload
    },
  },
})

export default editorPanelSlice.reducer

export const { setCaptionScore } = editorPanelSlice.actions

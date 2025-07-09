import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface EditorPanelState {
  captionScore: number | null
}

const initialState: EditorPanelState = {
  captionScore: null,
}

const editorPanelSlice = createSlice({
  name: 'editorPanel',
  initialState,
  reducers: {
    setCaptionScore: (state, action: PayloadAction<number | null>) => {
      state.captionScore = action.payload
    },
  },
})

export default editorPanelSlice.reducer

export const { setCaptionScore } = editorPanelSlice.actions

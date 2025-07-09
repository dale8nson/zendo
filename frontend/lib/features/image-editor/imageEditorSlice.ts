import { createSlice } from '@reduxjs/toolkit'
import { MetadataEntry } from '@/components/ImageGallery'

interface ImageEditorState {
  selectedImage: MetadataEntry | null
  caption: string | null
  editorCanvasData: string | null
  previewCanvasData: string | null
}

const initialState: ImageEditorState = {
  selectedImage: null,
  caption: null,
  editorCanvasData: null,
  previewCanvasData: null,
}

const imageEditorSlice = createSlice({
  name: 'imageEditor',
  initialState,
  reducers: {
    setSelectedImage(state, action) {
      state.selectedImage = action.payload
    },
    setCaption(state, action) {
      state.caption = action.payload
    },
    setEditorCanvasData(state, action) {
      state.editorCanvasData = action.payload
    },
    setPreviewCanvasData(state, action) {
      state.previewCanvasData = action.payload
    },
  },
})

export const { setSelectedImage, setCaption, setEditorCanvasData, setPreviewCanvasData } =
  imageEditorSlice.actions

export default imageEditorSlice.reducer

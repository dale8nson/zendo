import { createSlice } from '@reduxjs/toolkit'
import { MetadataEntry } from '@/components/ImageGallery'

interface ImageEditorState {
  image: MetadataEntry | null
}

const initialState: ImageEditorState = {
  image: null,
}

const imageEditorSlice = createSlice({
  name: 'imageEditor',
  initialState,
  reducers: {
    setSelectedImage(state, action) {
      state.image = action.payload
    },
    selectedImage(state, action) {
      return { ...state, image: state.image }
    },
  },
})

export const { setSelectedImage, selectedImage } = imageEditorSlice.actions

export default imageEditorSlice.reducer

import { createSlice } from '@reduxjs/toolkit'
import { MetadataEntry } from '@/components/ImageGallery'

interface ImageEditorState {
  image: MetadataEntry | null
  caption: string | null
}

const initialState: ImageEditorState = {
  image: null,
  caption: null,
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
    setCaption(state, action) {
      state.caption = action.payload
    },
    caption(state) {
      return { ...state, caption: state.caption }
    },
  },
})

export const { setSelectedImage, selectedImage, setCaption, caption } = imageEditorSlice.actions

export default imageEditorSlice.reducer

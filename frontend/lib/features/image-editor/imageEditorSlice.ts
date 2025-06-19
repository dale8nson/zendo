import { createSlice } from '@reduxjs/toolkit'
import { MetadataEntry } from '@/components/ImageGallery'

const imageEditorSlice = createSlice({
  name: 'imageEditor',
  initialState: {
    image: null,
  },
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

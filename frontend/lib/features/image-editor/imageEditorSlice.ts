import { createSlice } from '@reduxjs/toolkit'

interface ImageEditorState {
  selectedImage: MetadataEntry | null
  caption: string | null
  editorCanvasData: string | null
  status: {}
  maskData: MaskData[]
  maskIndex: { value: number }
  masks: { [key: string]: HTMLImageElement }[]
  selectedMaskData: MaskData[]
  selectedMasks: { id: string; imageData: string }[]
  collection: string
  token: string
  maskBox: number[]
  scaledSelectionBox: number[]
  objectCaption: string
  selectionBox: number[]
}

const initialState: ImageEditorState = {
  selectedImage: null,
  caption: null,
  editorCanvasData: null,
  status: {},
  maskData: [],
  maskIndex: { value: 0 },
  masks: [],
  selectedMaskData: [],
  collection: 'default',
  token: '',
  maskBox: [0, 0, 0, 0],
  scaledSelectionBox: [0, 0, 0, 0],
  objectCaption: '',
  selectedMasks: [],
  selectionBox: [0, 0, 0, 0],
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
    setEditorCanvasStatus(state, action) {
      state.status = action.payload
    },
    setMaskData: (state, action) => {
      state.maskData = action.payload
    },
    setMasks(state, action) {
      state.masks = action.payload
    },
    setMaskIndex(state, action) {
      state.maskIndex = { value: action.payload }
    },
    setSelectedMaskData(state, action) {
      state.selectedMaskData = action.payload
    },
    setCollection(state, action) {
      state.collection = action.payload
    },
    setToken(state, action) {
      state.token = action.payload
    },
    setMaskBox(state, action) {
      state.maskBox = action.payload
    },
    setSelectionBox(state, action) {
      state.selectionBox = action.payload
    },
    setScaledSelectionBox(state, action) {
      state.scaledSelectionBox = action.payload
    },
    setObjectCaption(state, action) {
      state.objectCaption = action.payload
    },
    setSelectedMasks(state, action) {
      state.selectedMasks = action.payload
    },
    nextMask(state) {
      let maskData = state.maskData
      const index = state.maskIndex.value
      maskData[index].active = false
      maskData[(index + 1) % maskData.length].active = true
      state.maskIndex = { value: (index + 1) % maskData.length }
      state.maskData = maskData
    },
    includeMask(state, action) {
      const index = action.payload
      const maskData = [...state.maskData]
      if (index < 0 || index >= maskData.length) return
      maskData[index].include = true
      maskData[index].exclude = false
      state.maskData = maskData
    },
    excludeMask(state, action) {
      const index = action.payload
      const maskData = [...state.maskData]
      if (index < 0 || index >= maskData.length) return
      maskData[index].include = false
      maskData[index].exclude = true
      state.maskData = maskData
    },
  },
})

export const {
  setSelectedImage,
  setCaption,
  setEditorCanvasData,
  setEditorCanvasStatus,
  setMaskData,
  setMasks,
  setMaskIndex,
  setSelectedMaskData,
  setCollection,
  setToken,
  setMaskBox,
  setSelectionBox,
  setScaledSelectionBox,
  setObjectCaption,
  setSelectedMasks,
  nextMask,
  includeMask,
  excludeMask,
} = imageEditorSlice.actions

export default imageEditorSlice.reducer

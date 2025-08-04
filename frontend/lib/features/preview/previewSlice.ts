import { createSlice } from '@reduxjs/toolkit'
//         list(dict(str, any)): A list over records for masks. Each record is
// a dict containing the following keys:
//   segmentation (dict(str, any) or np.ndarray): The mask. If
//     output_mode='binary_mask', is an array of shape HW. Otherwise,
//     is a dictionary containing the RLE.
//   bbox (list(float)): The box around the mask, in XYWH format.
//   area (int): The area in pixels of the mask.
//   predicted_iou (float): The model's own prediction of the mask's
//     quality. This is filtered by the pred_iou_thresh parameter.
//   point_coords (list(list(float))): The point coordinates input
//     to the model to generate this mask.
//   stability_score (float): A measure of the mask's quality. This
//     is filtered on using the stability_score_thresh parameter.
//   crop_box (list(float)): The crop of the image used to generate
//     the mask, given in XYWH format.

interface PreviewState {
  previewCanvasData: string
  shouldDrawCanvas: boolean
  shouldDrawMasks: boolean
  maskData: MaskData[]
  progress: number
  history: string[]
  currentHistoryIndex: number
  status: {}
  scaledSelectionBox: number[]
  maskIndex: { value: number }
  maskBox: number[]
  selectedMaskData: MaskData[]
  selectedMasks: SelectedMask[]
  layeredHistory: Layer[][]
}

const initialState: PreviewState = {
  previewCanvasData: '',
  shouldDrawCanvas: false,
  shouldDrawMasks: false,
  maskData: [],
  progress: 0,
  history: [],
  currentHistoryIndex: -1,
  status: {},
  scaledSelectionBox: [0, 0, 0, 0],
  maskIndex: { value: 0 },
  maskBox: [0, 0, 0, 0],
  selectedMaskData: [],
  selectedMasks: [],
  layeredHistory: [],
}

const previewSlice = createSlice({
  name: 'preview',
  initialState,
  reducers: {
    setPreviewCanvasData(state, action) {
      state.previewCanvasData = action.payload
    },
    setShouldDrawCanvas(state, action) {
      state.shouldDrawCanvas = action.payload
    },
    setShouldDrawMasks(state, action) {
      state.shouldDrawMasks = action.payload
    },
    setMaskData(state, action) {
      state.maskData = action.payload
    },
    setProgress(state, action) {
      state.progress = action.payload
    },
    appendHistory(state, action) {
      state.history = [...state.history, action.payload]
    },
    setCurrentHistoryIndex(state, action) {
      state.currentHistoryIndex = action.payload
    },
    setPreviewStatus(state, action) {
      state.status = action.payload
    },
    setScaledSelectionBox(state, action) {
      state.scaledSelectionBox = action.payload
    },
    setMaskIndex(state, action) {
      state.maskIndex = { value: action.payload }
    },
    setMaskBox(state, action) {
      state.maskBox = action.payload
    },
    setSelectedMaskData(state, action) {
      state.selectedMaskData = action.payload
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
  setPreviewCanvasData,
  setShouldDrawCanvas,
  setShouldDrawMasks,
  setMaskData,
  setProgress,
  appendHistory,
  setCurrentHistoryIndex,
  setPreviewStatus,
  setScaledSelectionBox,
  setMaskIndex,
  setMaskBox,
  setSelectedMaskData,
  setSelectedMasks,
  nextMask,
  includeMask,
  excludeMask,
} = previewSlice.actions

export default previewSlice.reducer

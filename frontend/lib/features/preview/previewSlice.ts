import { createSlice } from '@reduxjs/toolkit'
import { MetadataEntry } from '@/components/ImageGallery'

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

interface MaskData {
  segmentation: string
  bbox: [number, number, number, number]
  area: number
  predicted_iou: number
  point_coords: [[number]]
  stability_score: number
  crop_box: [number, number, number, number]
}

interface PreviewState {
  previewCanvasData: string | null
  shouldDrawCanvas: boolean
  shouldDrawMasks: boolean
  maskData: [MaskData] | null
  progress: number
  history: string[]
  currentHistoryIndex: { value: number }
}

const initialState: PreviewState = {
  previewCanvasData: null,
  shouldDrawCanvas: false,
  shouldDrawMasks: false,
  maskData: null,
  progress: 0,
  history: [],
  currentHistoryIndex: { value: -1 },
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
      state.currentHistoryIndex = { ...state.currentHistoryIndex, value: action.payload }
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
} = previewSlice.actions

export default previewSlice.reducer

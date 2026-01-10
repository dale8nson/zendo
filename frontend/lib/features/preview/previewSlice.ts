import { createSlice } from '@reduxjs/toolkit'
import { LucideGripHorizontal } from 'lucide-react'
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

type Layers = Layer[]

interface PreviewState {
  previewCanvasData: string
  previewCanvasSize: number[]
  shouldDrawCanvas: boolean
  shouldDrawMasks: boolean
  maskData: MaskData[]
  progress: number
  history: string[]
  currentHistoryIndex: number
  status: {}
  scaledSelectionBox: number[]
  maskIndex: number
  maskBox: number[]
  selectedMaskData: MaskData[]
  selectedMasks: SelectedMask[]
  layerHistory: Layers[]
  selectedLayer: Layer | null
  rootBbox: number[]
}

const initialState: PreviewState = {
  previewCanvasData: '',
  previewCanvasSize: [0, 0, 0, 0],
  shouldDrawCanvas: false,
  shouldDrawMasks: false,
  maskData: [],
  progress: 0,
  history: [],
  currentHistoryIndex: -1,
  status: {},
  scaledSelectionBox: [0, 0, 0, 0],
  maskIndex: 0,
  maskBox: [0, 0, 0, 0],
  selectedMaskData: [],
  selectedMasks: [],
  layerHistory: [],
  selectedLayer: null,
  rootBbox: [0, 0, 0, 0],
}

const previewSlice = createSlice({
  name: 'preview',
  initialState,
  reducers: {
    setPreviewCanvasData(state, action) {
      state.previewCanvasData = action.payload
    },
    setPreviewCanvasSize(state, action) {
      state.previewCanvasSize = action.payload
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
      state.maskIndex = action.payload
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
      let maskData = [...state.maskData]
      const index = state.maskIndex
      maskData[index].active = false
      maskData[(index + 1) % maskData.length].active = true
      state.maskIndex = (index + 1) % maskData.length
      state.maskData = [...maskData]
    },
    previousMask(state) {
      let maskData = [...state.maskData]
      const index = state.maskIndex
      maskData[index].active = false
      maskData[(index > 0 ? index : maskData.length) - 1].active = true
      state.maskIndex = (index > 0 ? index : maskData.length) - 1
      state.maskData = [...maskData]
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
    selectLayer(state, action) {
      let layerHistory = state.layerHistory
      layerHistory = layerHistory.map((layers, index) => {
        if (index === state.currentHistoryIndex) {
          return layers.map((layer, index) => {
            if (index === action.payload) {
              layer.selected = true
              state.selectedLayer = layer
            } else {
              layer.selected = false
            }
            return layer
          })
        }
      }) as Layers[]

      state.layerHistory = layerHistory
    },
    setLayerLabel(state, action) {
      let layerHistory = [...state.layerHistory]
      let layers = layerHistory[state.currentHistoryIndex]
      let layer = layers[action.payload.index]
      if (layer.label !== 'root' && action.payload.label !== 'root')
        layer.label = action.payload.label
      layerHistory[state.currentHistoryIndex][action.payload.index] = layer
      state.layerHistory = layerHistory
    },
    setLayerOpacity(state, action) {
      const { index, opacity } = action.payload
      let layerHistory = [...state.layerHistory]
      let layers = layerHistory[state.currentHistoryIndex]
      let layer = layers[index]
      layer.opacity = opacity
      layers[index] = layer
      layerHistory[state.currentHistoryIndex] = layers
      state.layerHistory = layerHistory
    },
    setLayerVisible(state, action) {
      const { index, visible } = action.payload
      let layerHistory = [...state.layerHistory]
      let layers = layerHistory[state.currentHistoryIndex]
      let layer = layers[index]
      layer.visible = visible
      layers[index] = layer
      layerHistory[state.currentHistoryIndex] = layers
      state.layerHistory = layerHistory
    },
    appendLayerHistory(state, action) {
      let layerHistory = state.layerHistory
      const { bbox, imageData } = action.payload
      layerHistory = layerHistory.map((layers, index) => {
        if (index === state.currentHistoryIndex) {
          return layers.map((layer, index) => {
            if (index === layer.currentLayerHistoryIndex) {
              layer.history.push({ bbox, imageData })
            }
            return layer
          })
        }
      }) as Layers[]

      state.layerHistory = layerHistory
    },
    setLayerHistoryIndex(state, action) {
      let layerHistory = state.layerHistory
      const { currentLayerHistoryIndex, imageData, bbox } = action.payload
      layerHistory = layerHistory.map((layers, index) => {
        if (index === state.currentHistoryIndex) {
          return layers.map((layer, index) => {
            if (index === currentLayerHistoryIndex) {
              layer.currentLayerHistoryIndex = currentLayerHistoryIndex
            }
            return layer
          })
        }
      }) as Layers[]

      state.layerHistory = layerHistory
    },
    newImage(state, action) {
      const { bbox, imageData } = action.payload

      const layer: Layer = {
        label: `root`,
        selected: true,
        visible: true,
        opacity: 1.0,
        currentLayerHistoryIndex: 0,
        history: [{ bbox, imageData }],
      }
      state.layerHistory = [...state.layerHistory, [layer]]
      state.currentHistoryIndex = state.layerHistory.length - 1
      state.selectedLayer = layer
      const index = layer.currentLayerHistoryIndex
      state.rootBbox = layer.history[index].bbox
    },
    newLayer(state, action) {
      const { bbox, imageData } = action.payload
      const layerHistory = [...state.layerHistory]
      const layers = layerHistory[state.currentHistoryIndex]
      const newLayer: Layer = {
        label: `layer-${layers.length}`,
        selected: true,
        visible: true,
        opacity: 1.0,
        currentLayerHistoryIndex: 0,
        history: [{ bbox, imageData }],
      }
      layers.push(newLayer)
      layerHistory[state.currentHistoryIndex] = layers
      state.selectedLayer = newLayer
    },
    newEmptyLayer(state) {
      const layer: Layer = {
        label: `root`,
        selected: false,
        visible: true,
        opacity: 1.0,
        currentLayerHistoryIndex: 0,
        history: [],
      }
      state.layerHistory[state.currentHistoryIndex].push(layer)
      state.selectedLayer = layer
    },
    updateLayer(state, action) {
      const { bbox, imageData } = action.payload
      let layerHistory = state.layerHistory

      layerHistory = layerHistory.map((layers, index) => {
        if (index == state.currentHistoryIndex) {
          return layers.map((layer) => {
            if (layer.selected) {
              layer.history.push({ bbox, imageData })
              layer.currentLayerHistoryIndex = layer.history.length - 1
            }
            return layer
          })
        }
      }) as Layers[]

      const layers = layerHistory[state.currentHistoryIndex]
      const layer = layers.find((layer) => layer.selected)
      if (layer) {
        layer.history.push({ bbox, imageData })
      }
    },
  },
})

export const {
  setPreviewCanvasData,
  setPreviewCanvasSize,
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
  selectLayer,
  setLayerLabel,
  setLayerOpacity,
  setLayerVisible,
  appendLayerHistory,
  newImage,
  newLayer,
  setLayerHistoryIndex,
  newEmptyLayer,
  updateLayer,
  previousMask,
} = previewSlice.actions

export default previewSlice.reducer

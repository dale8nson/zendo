declare interface Img2ImgRequest {
  layers: Layer[]
  prompt: string
  strength: number
  inference_steps: number
  guidance_scale: number
  negative_prompt: string
  prompt_2: string
  negative_prompt_2: string
  remove_background: boolean
}

declare interface InpaintRequest {
  layers: Layer[]
  prompt: string
  masks: MaskData[]
  strength: number
  inference_steps: number
  guidance_scale: number
  use_refiner: boolean
  inpaint_refiner_ratio: number
  inpaint_refiner_inference_steps: number
  inpaint_refiner_guidance_scale: number
  negative_prompt: string
  prompt_2: string
  negative_prompt_2: string
  refiner_prompt: string
  refiner_negative_prompt: string
  refiner_prompt_2: string
  refiner_negative_prompt_2: string
  new_layer: boolean
  remove_background: boolean
}

declare interface MasksRequest {
  image: string
}

declare interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[]
  data: TData[]
}

declare interface ScoreRequest {
  filename: string | undefined
  caption: string | null
}

declare interface CroppedImageMaskRequest {
  image: string
  bbox: number[]
}

declare interface CroppedImageCaptionRequest {
  image_data: string
  crop_box: number[]
}

declare interface MetadataEntry {
  id: number
  filename: string
  original_filename: string
  label: string | null
  prediction: string | null
  timestamp: string
  image_data: string
  width: number
  height: number
  collection: string
}

declare type Layer = {
  selected: boolean
  label: string
  visible: boolean
  opacity: number
  currentLayerHistoryIndex: number
  history: { bbox: []; imageData: string }[]
}

declare type TableLayerRowData = {
  selected: boolean
  label: string
  visible: boolean
  opacity: number
}

declare interface MaskData {
  id: string
  segmentation: string
  bbox: [number, number, number, number]
  area: number
  predicted_iou: number
  point_coords: number[]
  stability_score: number
  crop_box: [number, number, number, number]
  mask: string
  inverted_mask: string
  label: string
  active: boolean
  include: boolean
  exclude: boolean
  canvas_box: number[]
}

declare type Mask = {
  label: string
  active: boolean
  include: boolean
  exclude: boolean
}

declare type SelectedMask = {
  id: string
  imageData: string
}

declare type MaskImage = {
  segmentation: HTMLImageElement
  mask: HTMLImageElement
  inverted_mask: HTMLImageElement
}

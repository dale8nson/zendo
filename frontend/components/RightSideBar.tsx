'use client'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import {
  setGenerationIterations,
  setGenerationGuidanceScale,
  setImg2imgStrength,
  setImg2imgInferenceSteps,
  setImg2imgGuidanceScale,
  setImg2imgUseRefiner,
  setImg2imgRefinerRatio,
  setImg2imgRefinerInferenceSteps,
  setImg2imgRefinerGuidanceScale,
  setInpaintStrength,
  setInpaintInferenceSteps,
  setInpaintGuidanceScale,
  setInpaintUseRefiner,
  setInpaintRefinerRatio,
  setInpaintRefinerInferenceSteps,
  setInpaintRefinerGuidanceScale,
  toggleDisabled,
} from '@/lib/features/control-panel/controlPanelSlice'
import { MouseEvent, MouseEventHandler, useEffect, useRef, useState } from 'react'
import { Button } from './Button'
import { LayerTable } from './LayerTable'
import {
  setMaskData,
  setProgress,
  appendHistory,
  setCurrentHistoryIndex,
  setPreviewStatus,
  newImage,
  selectLayer,
  setLayerOpacity,
  setLayerVisible,
  appendLayerHistory,
  newLayer,
} from '@/lib/features/preview/previewSlice'

import { MaskTable } from '@/components/MaskTable'

import { rustUpscale, sdxlGenerateViaWs, backendUrl } from '@/lib/backend'

const generate = async (
  prompt: string,
  iterations: number,
  guidance_scale: number,
  negative_prompt: string,
  prompt_2: string,
  negative_prompt_2: string,
  ip_adapter_image: string | Layer[] | null,
  use_face_id: boolean,
  bbox: number[] | null,
  remove_background: boolean,
  use_ip_adapter_image: boolean,
  refiner_strength: number,
  seed: number | null,
  threshold1: number,
  threshold2: number,
  aperture_size: number,
  l2_gradient: number,
  onProgress?: (step: number, total: number) => void
): Promise<string> => {
  // Map UI sliders to SDXL params
  const steps = Math.max(1, Math.min(50, iterations || 25))
  const gs = Number.isFinite(guidance_scale) ? guidance_scale : 7.5
  const width = 1024
  const height = 1024
  const params = {
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    guidance_scale: gs,
    seed: seed ?? undefined,
  }
  return await sdxlGenerateViaWs(params, onProgress)
}

const get_masks = async (request: MasksRequest) => {
  console.log(`request: ${JSON.stringify(request)}`)
  const response = await fetch('http://localhost:8000/api/masks', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    cache: 'no-cache',
    body: JSON.stringify(request),
  })
  const data = await response.json()
  return data
}

export function RightSideBar() {
  console.log('RightSideBar')

  const layerHistory = useAppSelector((state) => state.preview.layerHistory)

  const generationIterations = useAppSelector((state) => state.controlPanel.generationIterations)
  const generationGuidanceScale = useAppSelector(
    (state) => state.controlPanel.generationGuidanceScale
  )

  const caption = useAppSelector((state) => state.imageEditor.caption)
  const history = useAppSelector((state) => state.preview.history)
  const currentHistoryIndex = useAppSelector((state) => state.preview.currentHistoryIndex)
  const generatePrompt = useAppSelector((state) => state.promptPanel.generate_prompt)
  const generateNegativePrompt = useAppSelector(
    (state) => state.promptPanel.generate_negativePrompt
  )
  const generatePrompt2 = useAppSelector((state) => state.promptPanel.generate_prompt2)
  const generateNegativePrompt2 = useAppSelector(
    (state) => state.promptPanel.generate_negativePrompt2
  )
  const img2imgStrength = useAppSelector((state) => state.controlPanel.img2imgStrength)
  const img2imgInferenceSteps = useAppSelector((state) => state.controlPanel.img2imgInferenceSteps)
  const img2imgGuidanceScale = useAppSelector((state) => state.controlPanel.img2imgGuidanceScale)
  const img2imgUseRefiner = useAppSelector((state) => state.controlPanel.img2imgUseRefiner)
  const img2imgRefinerRatio = useAppSelector((state) => state.controlPanel.img2imgRefinerRatio)
  const img2imgRefinerInferenceSteps = useAppSelector(
    (state) => state.controlPanel.img2imgRefinerInferenceSteps
  )
  const img2imgRefinerGuidanceScale = useAppSelector(
    (state) => state.controlPanel.img2imgRefinerGuidanceScale
  )
  const img2imgPrompt = useAppSelector((state) => state.promptPanel.img2img_prompt)
  const img2imgNegativePrompt = useAppSelector((state) => state.promptPanel.img2img_negativePrompt)
  const img2imgPrompt2 = useAppSelector((state) => state.promptPanel.img2img_prompt2)
  const img2imgNegativePrompt2 = useAppSelector(
    (state) => state.promptPanel.img2img_negativePrompt2
  )
  const inpaintPrompt = useAppSelector((state) => state.promptPanel.inpaint_prompt)
  const inpaintStrength = useAppSelector((state) => state.controlPanel.inpaintStrength)
  const inpaintInferenceSteps = useAppSelector((state) => state.controlPanel.inpaintInferenceSteps)
  const inpaintGuidanceScale = useAppSelector((state) => state.controlPanel.inpaintGuidanceScale)
  const inpaintUseRefiner = useAppSelector((state) => state.controlPanel.inpaintUseRefiner)
  const inpaintRefinerRatio = useAppSelector((state) => state.controlPanel.inpaintRefinerRatio)
  const inpaintRefinerInferenceSteps = useAppSelector(
    (state) => state.controlPanel.inpaintRefinerInferenceSteps
  )
  const inpaintRefinerGuidanceScale = useAppSelector(
    (state) => state.controlPanel.inpaintRefinerGuidanceScale
  )
  const inpaintNegativePrompt = useAppSelector((state) => state.promptPanel.inpaint_negativePrompt)
  const inpaintPrompt2 = useAppSelector((state) => state.promptPanel.inpaint_prompt2)
  const inpaintNegativePrompt2 = useAppSelector(
    (state) => state.promptPanel.inpaint_negativePrompt2
  )

  const selectedMasks = useAppSelector((state) => state.preview.selectedMasks)
  const maskIndex = useAppSelector((state) => state.preview.maskIndex)
  const disabled = useAppSelector((state) => state.controlPanel.disabled)
  const alpha = useAppSelector((state) => state.controlPanel.alpha)
  const maskData = useAppSelector((state) => state.preview.maskData)
  const previewCanvasData = useAppSelector((state) => state.preview.previewCanvasData)
  const previewStatus = useAppSelector((state) => state.preview.status)
  const selectedImage = useAppSelector((state) => state.imageEditor.selectedImage)
  const editorSelectionBox = useAppSelector((state) => state.imageEditor.selectionBox)
  const editorScaledSelectionBox = useAppSelector((state) => state.imageEditor.scaledSelectionBox)

  const dispatch = useAppDispatch()

  const [useIPAdapterImage, setUseIPAdapterImage] = useState(false)
  const [useIPAdapterFaceID, setUseIPAdapterFaceID] = useState(false)
  const [toNewLayer, setToNewLayer] = useState(false)
  const [activeLayer, setActiveLayer] = useState(0)
  const [reuseCurrentImage, setReuseCurrentImage] = useState(false)
  const [generateRemoveBG, setGenerateRemoveBG] = useState(false)
  const [img2imgRemoveBG, setImg2imgRemoveBG] = useState(false)
  const [inpaintRemoveBG, setInpaintRemoveBG] = useState(false)
  const [generateRefinerStrength, setGenerateRefinerStrength] = useState(0.8)
  const [generateSeed, setGenerateSeed] = useState(0)
  const [generateUseSeed, setGenerateUseSeed] = useState(false)
  const [generateThreshold1, setGenerateThreshold1] = useState(100)
  const [generateThreshold2, setGenerateThreshold2] = useState(200)
  const [generateApertureSize, setGenerateApertureSize] = useState(3)
  const [generateL2Gradient, setGenerateL2Gradient] = useState(false)

  const scratchPad = useRef(null)

  const generateButtonHandler = async () => {
    console.log('Button clicked')
    if (!prompt) return

    const [x, y, w, h] = editorScaledSelectionBox
    ;(() => console.log(`editorScaledSelectionBox`, editorScaledSelectionBox))()
    const bbox = [x, y, x + w, y + h]
    let adapterImage: string | Layer[] | null = null
    if (useIPAdapterImage) {
      adapterImage = selectedImage?.image_data as string
    }
    if (reuseCurrentImage) {
      adapterImage = layerHistory[currentHistoryIndex] as Layer[]
    }

    ;(() => console.log(`adapterImage`, adapterImage))()

    dispatch(toggleDisabled(true))
    dispatch(setProgress(0))
    const b64 = await generate(
      generatePrompt,
      generationIterations,
      generationGuidanceScale,
      generateNegativePrompt,
      generatePrompt2,
      generateNegativePrompt2,
      adapterImage,
      useIPAdapterFaceID,
      bbox,
      generateRemoveBG,
      useIPAdapterImage,
      generateRefinerStrength,
      generateUseSeed? generateSeed : null,
      generateThreshold1,
      generateThreshold2,
      generateApertureSize,
      generateL2Gradient,
      (step, total) => {
        const pct = Math.round((step / total) * 100)
        dispatch(setProgress(pct))
      }
    )
    ;(() => console.log('Received image data'))()
    dispatch(toggleDisabled(false))
    ;(() => console.log('Dispatching new image'))()
    dispatch(newImage({ bbox: [0, 0, 1024, 1024], imageData: b64 }))
  }

  const upscale = async () => {
    const layers = layerHistory[currentHistoryIndex]
    const root = layers.find((layer) => layer.label === 'root') as Layer
    const index = root?.currentLayerHistoryIndex as number
    const img = root?.history?.[index]?.imageData as string | undefined
    if (!img) return
    try {
      const out = await rustUpscale(img)
      dispatch(newImage({ bbox: [0,0,1024,1024], imageData: out }))
    } catch (e) {
      console.error('Upscale error', e)
    }
  }

  const img2imgButtonHandler = async () => {
    console.log('img2img button clicked')
    if (!layerHistory[currentHistoryIndex]) return
    const oldStatus = previewStatus
    dispatch(
      setPreviewStatus({ Status: 'running img2img...', currentHistoryIndex: currentHistoryIndex })
    )
    const ws = new WebSocket('ws://localhost:8001/api/refine')
    const layers = layerHistory[currentHistoryIndex]
    const request: Img2ImgRequest = {
      prompt: img2imgPrompt,
      layers: layers,
      strength: img2imgStrength,
      inference_steps: img2imgInferenceSteps,
      guidance_scale: img2imgGuidanceScale,
      negative_prompt: img2imgNegativePrompt,
      prompt_2: img2imgPrompt2,
      negative_prompt_2: img2imgNegativePrompt2,
      remove_background: img2imgRemoveBG,
    }

    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      dispatch(toggleDisabled(false))
      ws.close()
      // dispatch(setPreviewStatus(oldStatus))
    }
    ws.onclose = (event) => {
      console.log('WebSocket closed', event)
      dispatch(toggleDisabled(false))
      // dispatch(setPreviewStatus(oldStatus))
    }
    ws.onopen = (event) => {
      console.log('WebSocket connection opened')
      ws.send(JSON.stringify(request))
      dispatch(toggleDisabled(true))
      dispatch(setProgress(0))
      ws.onmessage = (message) => {
        console.log(`message keys: `, Object.keys(message))
        const data = JSON.parse(message.data)
        console.log(data)
        if ('image_data' in data) {
          console.log(`data keys: ${Object.keys(data)}`)
          dispatch(toggleDisabled(false))
          dispatch(newImage({ bbox: [0, 0, 1024, 1024], imageData: data.image_data }))
          ws.close()
        } else if ('step' in data) {
          // Progress message!
          console.log(`Progress: step ${data.step}, t=${data.timestep}`)
        } else {
          console.log('WebSocket message received:', data)
        }
      }
    }
  }

  const inpaintWS = () => {
    if (!layerHistory[currentHistoryIndex]) return
    const oldStatus = previewStatus
    dispatch(
      setPreviewStatus({ Status: 'running Inpaint...', currentHistoryIndex: currentHistoryIndex })
    )
    const ws = new WebSocket('ws://localhost:8001/api/inpaint')
    const masks = maskData.filter((mask) => mask.include || mask.exclude)
    const request: InpaintRequest = {
      layers: layerHistory[currentHistoryIndex],
      prompt: inpaintPrompt,
      masks: masks,
      strength: inpaintStrength,
      inference_steps: inpaintInferenceSteps,
      guidance_scale: inpaintGuidanceScale,
      use_refiner: inpaintUseRefiner,
      inpaint_refiner_ratio: inpaintRefinerRatio,
      inpaint_refiner_inference_steps: inpaintRefinerInferenceSteps,
      inpaint_refiner_guidance_scale: inpaintRefinerGuidanceScale,
      negative_prompt: inpaintNegativePrompt,
      prompt_2: inpaintPrompt2,
      negative_prompt_2: inpaintNegativePrompt2,
      refiner_prompt: img2imgPrompt,
      refiner_negative_prompt: img2imgNegativePrompt,
      refiner_prompt_2: img2imgPrompt2,
      refiner_negative_prompt_2: img2imgNegativePrompt2,
      new_layer: toNewLayer,
      remove_background: inpaintRemoveBG,
    }

    dispatch(setProgress(0))
    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      ws.close()
      dispatch(setPreviewStatus(oldStatus))
      dispatch(toggleDisabled(false))
    }
    ws.onclose = (event) => {
      console.log('WebSocket closed', event)
      dispatch(setPreviewStatus(oldStatus))
      dispatch(toggleDisabled(false))
    }
    ws.onopen = (event) => {
      console.log('WebSocket connection opened')
      ws.send(JSON.stringify(request))
      dispatch(toggleDisabled(true))
      ws.onmessage = (message) => {
        console.log(`message keys: `, Object.keys(message))
        const data = JSON.parse(message.data)
        console.log(`message.data: ${message.data}`)
        console.log(data)
        if ('image_data' in data) {
          if (toNewLayer) {
            // const initial_bbox = masks[0].bbox
            // let bbox = masks.reduce((acc, mask) => {
            //   const [ax1, ay1, ax2, ay2] = acc
            //   const [mx1, my1, mx2, my2] = mask.bbox
            //   const x1 = Math.min(ax1, mx1)
            //   const y1 = Math.min(ay1, my1)
            //   const x2 = Math.max(ax2, mx2)
            //   const y2 = Math.max(ay2, my2)

            // return [x1, y1, x2, y2]
            // }, initial_bbox)
            const activeLayer = layerHistory[currentHistoryIndex].find(
              (layer) => layer.selected
            ) as Layer

            dispatch(newLayer({ bbox: data.bbox, imageData: data.image_data }))
          } else {
            dispatch(newImage({ bbox: [0, 0, 1024, 1024], imageData: data.image_data }))
          }

          let newMaskData = []
          for (const data of maskData) {
            let newData = { ...data }
            let bbox = newData.bbox
            const [x1, y1, x2, y2] = bbox
            const [w, h] = [x2 - x1, y2 - y1]
            const size = Math.max(w, h)
            const [mx, my] = [Math.floor((size - w) / 2), Math.floor((size - h) / 2)]
            // bbox = [x1 + mx, y1 + my, x2 + mx, y2 + my]
            let scaleX = size / w
            let scaleY = size / h
            const scale = Math.min(scaleX, scaleY)

            bbox = bbox.map((n) => n * scale) as [number, number, number, number]

            newData.bbox = bbox

            newMaskData.push(newData)
          }

          dispatch(setMaskData(newMaskData))

          dispatch(toggleDisabled(false))
          ws.close()
        } else if ('step' in data) {
          console.log(`Progress: step ${data.step}, t=${data.timestep}`)
        } else {
          console.log(`data keys: ${Object.keys(data)}`)
        }
      }
    }
  }

  useEffect(() => {
    if (!scratchPad.current) return
    const sp = scratchPad.current as HTMLTextAreaElement
    sp.value = localStorage.getItem('scratchPad') || ''
  }, [])

  return (
    <aside className="flex flex-col space-y-2 items-center justify-start w-full h-full p-4">
      <div className="flex flex-col items-start justify-start w-full h-[2.5rem]"></div>
      <div className="flex flex-col justify-start items-center overflow-y-scroll w-full h-full">
        <Accordion
          type="multiple"
          collapsible="true"
          className="w-full h-full space-y-2 overflow-y-scroll"
        >
          <AccordionItem value="image">
            <AccordionTrigger position="left">Image</AccordionTrigger>
            <AccordionContent className="flex flex-col gap-2">
              <Button
                onClick={() => {
                  if (disabled) return
                  upscale()
                }}
                // disabled={disabled || generatePrompt == ''}
              >
                Upscale
              </Button>
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-1">
            <AccordionTrigger position="left">Generation</AccordionTrigger>
            <AccordionContent className="flex flex-col gap-2">
              <label htmlFor="generationIterations">Iterations</label>
              <input
                id="generationIterations"
                type="text"
                value={generationIterations}
                onChange={(e) => dispatch(setGenerationIterations(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={generationIterations}
                onInput={(e) => dispatch(setGenerationIterations(Number(e.target.value)))}
              />
              <label htmlFor="generationGuidanceScale">Guidance Scale</label>
              <input
                id="generationGuidanceScale"
                type="text"
                value={generationGuidanceScale}
                onChange={(e) => dispatch(setGenerationGuidanceScale(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={0.25}
                max={100}
                step={0.25}
                value={generationGuidanceScale}
                onChange={(e) => dispatch(setGenerationGuidanceScale(Number(e.target.value)))}
              />
              <label htmlFor="generateRefinerStrength">Refiner Strength</label>
              <input
                id="generateRefinerStrength"
                type="text"
                value={generateRefinerStrength}
                onChange={(e) => setGenerateRefinerStrength(Number(e.target.value))}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={0.0}
                max={1.0}
                step={0.025}
                value={generateRefinerStrength}
                onChange={(e) => setGenerateRefinerStrength(Number(e.target.value))}
              />
              <label htmlFor="generateSeed">Seed</label>
              <div className='flex'>
              <input
                  id="generateUseSeed"
                  type="checkbox"
                  checked={generateUseSeed}
                  onChange={(e) => setGenerateUseSeed(e.target.checked)}
                />
              <input
                id="generateRefinerStrength"
                type="text"
                value={generateSeed}
                onChange={(e) => setGenerateSeed(Number(e.target.value))}
                className="w-full bg-white text-black p-2 font-bold"
                disabled={!generateUseSeed}
              />
              </div>
              <div className="flex items-center space-x-2">
                <input
                  id="useIPAdapterImage"
                  type="checkbox"
                  checked={useIPAdapterImage}
                  onChange={(e) => setUseIPAdapterImage(e.target.checked)}
                />
                <label htmlFor="useIPAdapterImage">IP Adapter Image</label>
                <input
                  id="useIPAdapterFaceID"
                  type="checkbox"
                  checked={reuseCurrentImage}
                  onChange={(e) => setReuseCurrentImage(e.target.checked)}
                />
                <label htmlFor="reuseCurrentImage">Reuse Current Image</label>
                <input
                  id="useIPAdapterFaceID"
                  type="checkbox"
                  checked={useIPAdapterFaceID}
                  onChange={(e) => setUseIPAdapterFaceID(e.target.checked)}
                />
                <label htmlFor="useIPAdapterFaceID">IP Adapter FaceID</label>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  id="inpaintRemoveBG"
                  type="checkbox"
                  checked={generateRemoveBG}
                  onChange={(e) => setGenerateRemoveBG(e.target.checked)}
                />
                <label htmlFor="generateRemoveBG">Remove Background</label>
              </div>
              <div className="flex flex-col space-y-4">
                <label htmlFor='generateThreshold1'>Canny Threshold 1</label>
                <input 
                  id='generateThreshold1' 
                  type='text' 
                  className='text-black bg-white' 
                  value={generateThreshold1.toString()} 
                  onChange={(e) => setGenerateThreshold1(Number(e.target.value))} 
                />
                <input 
                  type='range' 
                  value={generateThreshold1} 
                  min={0} 
                  max={1000} 
                  step={1} 
                  onChange={e => setGenerateThreshold1(Number(e.target.value))} 
                />
                <label htmlFor='generateThreshold2'>Canny Threshold 2</label>
                <input 
                  id='generateThreshold2' 
                  type='text' 
                  className='text-black bg-white' 
                  value={generateThreshold2.toString()} 
                  onChange={(e) => setGenerateThreshold2(Number(e.target.value))} 
                />
                <input 
                  type='range' 
                  value={generateThreshold2} 
                  min={0} 
                  max={1000} 
                  step={1} 
                  onChange={e => setGenerateThreshold2(Number(e.target.value))} 
                />
                <label htmlFor='generateApertureSize'>Aperture Size</label>
                <input 
                  id='generateApertureSize' 
                  type='text' 
                  className='text-black bg-white' 
                  value={generateApertureSize.toString()} 
                  onChange={(e) => setGenerateApertureSize(Number(e.target.value))} 
                />
                <input 
                  type='range' 
                  value={generateApertureSize} 
                  min={3} 
                  max={7} 
                  step={2} 
                  onChange={e => setGenerateApertureSize(Number(e.target.value))} 
                />
                <div className='flex w-full'>
                  <input 
                    id='generateL2Gradient' 
                    type='checkbox' 
                    checked={generateL2Gradient} 
                    onChange={e => setGenerateL2Gradient(e.target.checked)} 
                  />
                  <label htmlFor='generateL2Gradient' className='text-white'>Use L2 Gradient</label>
                </div>
              </div>
              <Button
                onClick={() => {
                  if (disabled) return

                  generateButtonHandler()
                }}
                // disabled={disabled || generatePrompt == ''}
              >
                Generate
              </Button>
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-2">
            <AccordionTrigger position="left">Img2Img</AccordionTrigger>
            <AccordionContent className="flex flex-col gap-2">
              <label htmlFor="img2imgStrength">Strength</label>
              <input
                id="img2imgStrength"
                type="text"
                value={img2imgStrength.toString()}
                onChange={(e) => dispatch(setImg2imgStrength(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={0.02}
                max={1}
                step={0.02}
                value={img2imgStrength}
                onChange={(e) => dispatch(setImg2imgStrength(Number(e.target.value)))}
              />
              <label htmlFor="img2imgInferenceSteps">Inference Steps</label>
              <input
                type="text"
                id="img2imgInferenceSteps"
                value={img2imgInferenceSteps.toString()}
                onChange={(e) => dispatch(setImg2imgInferenceSteps(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={img2imgInferenceSteps}
                onChange={(e) => dispatch(setImg2imgInferenceSteps(Number(e.target.value)))}
              />
              <label htmlFor="img2imgRefinerGuidanceScale">Guidance Scale</label>
              <input
                id="img2imgGuidanceScale"
                type="text"
                value={img2imgGuidanceScale.toString()}
                onChange={(e) => dispatch(setImg2imgGuidanceScale(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={1}
                max={20}
                step={0.25}
                value={img2imgGuidanceScale}
                onChange={(e) => dispatch(setImg2imgGuidanceScale(Number(e.target.value)))}
              />
              <div className="flex items-center space-x-2">
                <input
                  id="img2imgRemoveBG"
                  type="checkbox"
                  checked={img2imgRemoveBG}
                  onChange={(e) => setImg2imgRemoveBG(e.target.checked)}
                />
                <label htmlFor="img2imgRemoveBG">Remove Background</label>
              </div>
              <Button
                onClick={() => {
                  if (disabled) return
                  img2imgButtonHandler()
                }}
                disabled={disabled || !layerHistory[currentHistoryIndex]}
              >
                Img2Img
              </Button>
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-3">
            <AccordionTrigger position="left">Inpaint</AccordionTrigger>
            <AccordionContent className="flex flex-col gap-2">
              <label htmlFor="inpaintStrength">Strength</label>
              <input
                id="inpaintStrength"
                type="text"
                value={inpaintStrength.toString()}
                onChange={(e) => dispatch(setInpaintStrength(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={0.02}
                max={1}
                step={0.02}
                value={inpaintStrength}
                onChange={(e) => dispatch(setInpaintStrength(Number(e.target.value)))}
              />
              <label htmlFor="inpaintInferenceSteps">Inference Steps</label>
              <input
                type="text"
                id="inpaintInferenceSteps"
                value={inpaintInferenceSteps.toString()}
                onChange={(e) => dispatch(setInpaintInferenceSteps(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
                disabled={!inpaintUseRefiner}
              />
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={inpaintInferenceSteps}
                onChange={(e) => dispatch(setInpaintInferenceSteps(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <label htmlFor="inpaintGuidanceScale">Guidance Scale</label>
              <input
                id="inpaintGuidanceScale"
                type="text"
                value={inpaintGuidanceScale.toString()}
                onChange={(e) => dispatch(setInpaintGuidanceScale(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={0}
                max={100}
                step={0.25}
                value={inpaintGuidanceScale}
                onChange={(e) => dispatch(setInpaintGuidanceScale(Number(e.target.value)))}
              />
              <div className="flex items-center space-x-2">
                <input
                  id="inpaintUseRefiner"
                  type="checkbox"
                  checked={inpaintUseRefiner}
                  onChange={(e) => dispatch(setInpaintUseRefiner(e.target.checked))}
                />
                <label htmlFor="inpaintUseRefiner">Refiner</label>
              </div>
              <label htmlFor="inpaintRefinerRatio">Inpaint/Refiner Ratio</label>
              <input
                id="inpaintRefinerRatio"
                type="text"
                value={inpaintRefinerRatio}
                onChange={(e) => dispatch(setInpaintRefinerRatio(Number(e.target.value)))}
                disabled={!inpaintUseRefiner}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={0}
                max={1.0}
                step={0.05}
                value={inpaintRefinerRatio}
                onChange={(e) => dispatch(setInpaintRefinerRatio(Number(e.target.value)))}
                disabled={!inpaintUseRefiner}
              />
              <label htmlFor="inpaintRefinerInferenceSteps">Refiner Inference Steps</label>
              <input
                type="text"
                value={inpaintRefinerInferenceSteps}
                onChange={(e) => dispatch(setInpaintRefinerInferenceSteps(Number(e.target.value)))}
                disabled={!inpaintUseRefiner}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={inpaintRefinerInferenceSteps}
                onChange={(e) => dispatch(setInpaintRefinerInferenceSteps(Number(e.target.value)))}
                disabled={!inpaintUseRefiner}
              />
              <label htmlFor="inpaintRefinerGuidanceScale">Refiner Guidance Scale</label>
              <input
                type="text"
                value={inpaintRefinerGuidanceScale}
                onChange={(e) => dispatch(setInpaintRefinerGuidanceScale(Number(e.target.value)))}
                disabled={!inpaintUseRefiner}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <input
                type="range"
                min={0}
                max={20}
                step={0.25}
                value={inpaintRefinerGuidanceScale}
                onChange={(e) => dispatch(setInpaintRefinerGuidanceScale(Number(e.target.value)))}
                disabled={!inpaintUseRefiner}
                className="w-full bg-white text-black p-2 font-bold"
              />
              <div className="flex items-center space-x-2">
                <input
                  id="toNewLayer"
                  type="checkbox"
                  checked={toNewLayer}
                  onChange={(e) => setToNewLayer(e.target.checked)}
                />
                <label htmlFor="toNewLayer">To New Layer</label>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  id="inpaintRemoveBG"
                  type="checkbox"
                  checked={inpaintRemoveBG}
                  onChange={(e) => setInpaintRemoveBG(e.target.checked)}
                />
                <label htmlFor="inpaintRemoveBG">Remove Background</label>
              </div>
              <Button
                onClick={() => {
                  if (disabled) return
                  inpaintWS()
                }}
                disabled={disabled || !maskData.length}
              >
                Inpaint
              </Button>
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="scratch">
            <AccordionTrigger position="left">Scratch</AccordionTrigger>
            <AccordionContent>
              <textarea
                ref={scratchPad}
                className="w-full h-16 p-2 border border-gray-300 rounded-md"
                onChange={(e) => localStorage.setItem('scratchPad', e.target.value)}
              />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
        <LayerTable />
      </div>
    </aside>
  )
}

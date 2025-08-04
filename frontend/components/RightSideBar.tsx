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
} from '@/lib/features/preview/previewSlice'

import { MaskTable } from '@/components/MaskTable'

const generate = async (
  prompt: string,
  iterations: number,
  guidance_scale: number,
  negative_prompt: string,
  prompt_2: string,
  negative_prompt_2: string
): Promise<string> => {
  console.log('getPreview called with prompt:', prompt)
  const response = await fetch('http://localhost:8001/api/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt,
      iterations,
      guidance_scale,
      negative_prompt,
      prompt_2,
      negative_prompt_2,
    }),
    // keepalive: true,
    cache: 'no-cache',
  })
  const data = await response.json()
  return data.image
}

// const img2img = async (request: Img2ImgRequest) => {
//   const response = await fetch('http://localhost:8001/api/refine', {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     cache: 'no-cache',
//     body: JSON.stringify(request),
//   })
//   const data = await response.json()
//   return data.image_data
// }

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

  const dispatch = useAppDispatch()

  const scratchPad = useRef(null)

  const generateButtonHandler = async () => {
    console.log('Button clicked')
    if (!prompt) return

    const b64 = await generate(
      generatePrompt,
      generationIterations,
      generationGuidanceScale,
      generateNegativePrompt,
      generatePrompt2,
      generateNegativePrompt2
    )
    console.log(`b64: ${b64.slice(0, 29)}`)
    dispatch(appendHistory(b64))
    dispatch(setCurrentHistoryIndex(history.length))
    dispatch(toggleDisabled(false))
  }

  const img2imgButtonHandler = async () => {
    console.log('img2img button clicked')
    if (!caption || !history[currentHistoryIndex]) return
    const oldStatus = previewStatus
    dispatch(
      setPreviewStatus({ Status: 'running img2img...', currentHistoryIndex: currentHistoryIndex })
    )
    const ws = new WebSocket('ws://localhost:8001/api/refine')
    const request: Img2ImgRequest = {
      prompt: img2imgPrompt,
      image: history[currentHistoryIndex] as string,
      strength: img2imgStrength,
      inference_steps: img2imgInferenceSteps,
      guidance_scale: img2imgGuidanceScale,
      negative_prompt: img2imgNegativePrompt,
      prompt_2: img2imgPrompt2,
      negative_prompt_2: img2imgNegativePrompt2,
    }

    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      ws.close()
      dispatch(toggleDisabled(false))
      dispatch(setPreviewStatus(oldStatus))
    }
    ws.onclose = (event) => {
      console.log('WebSocket closed', event)
      dispatch(toggleDisabled(false))
      dispatch(setPreviewStatus(oldStatus))
    }
    ws.onopen = (event) => {
      console.log('WebSocket connection opened')
      ws.send(JSON.stringify(request))

      dispatch(setProgress(0))
      ws.onmessage = (message) => {
        console.log(`message keys: `, Object.keys(message))
        const data = JSON.parse(message.data)
        console.log(data)
        if ('image_data' in data) {
          console.log(`data keys: ${Object.keys(data)}`)
          dispatch(appendHistory(data.image_data))
          dispatch(setCurrentHistoryIndex(history.length))
          dispatch(setPreviewStatus(oldStatus))
          dispatch(toggleDisabled(false))
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
    if (!history[currentHistoryIndex]) return
    const oldStatus = previewStatus
    dispatch(
      setPreviewStatus({ Status: 'running Inpaint...', currentHistoryIndex: currentHistoryIndex })
    )
    const ws = new WebSocket('ws://localhost:8001/api/inpaint')
    const masks = maskData.filter((mask) => mask.include || mask.exclude)
    const request: InpaintRequest = {
      image: history[currentHistoryIndex] as string,
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
      ws.onmessage = (message) => {
        console.log(`message keys: `, Object.keys(message))
        const data = JSON.parse(message.data)
        console.log(`message.data: ${message.data}`)
        console.log(data)
        if ('image_data' in data) {
          dispatch(appendHistory(data.image_data))
          dispatch(setCurrentHistoryIndex(history.length))
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
                min={0}
                max={100}
                step={0.25}
                value={generationGuidanceScale}
                onChange={(e) => dispatch(setGenerationGuidanceScale(Number(e.target.value)))}
              />
              <Button
                onClick={() => {
                  if (disabled) return
                  dispatch(toggleDisabled(true))
                  generateButtonHandler()
                }}
                disabled={disabled || generatePrompt == ''}
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
              <Button
                onClick={() => {
                  if (disabled) return
                  dispatch(toggleDisabled(true))
                  img2imgButtonHandler()
                }}
                disabled={disabled || !history[currentHistoryIndex]}
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
              <Button
                onClick={() => {
                  if (disabled) return
                  dispatch(toggleDisabled(true))
                  inpaintWS()
                }}
                disabled={disabled || !maskData}
              >
                Inpaint
              </Button>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
        {/*<LayerTable />*/}
        <textarea
          ref={scratchPad}
          className="w-full h-16 p-2 border border-gray-300 rounded-md"
          onChange={(e) => localStorage.setItem('scratchPad', e.target.value)}
        />
      </div>
    </aside>
  )
}

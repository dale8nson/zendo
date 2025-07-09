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
  setImg2imgGuidanceScale,
  setRefinementStrength,
  setRefinementGuidanceScale,
  setInpaintStrength,
  setInpaintGuidanceScale,
  setMasked,
  setAlpha,
  setMaskIndex,
  toggleMaskVisible,
  toggleDisabled,
  setInpaintNoise,
  setInpaintNoiseOffset,
  setInpaintBlur,
  toggleStrict,
  toggleReverseMask,
  toggleUseNoise,
} from '@/lib/features/control-panel/controlPanelSlice'
import { MouseEventHandler, useRef, useState } from 'react'
import { Button } from './Button'
import {
  setPreviewCanvasData,
  setShouldDrawCanvas,
  setShouldDrawMasks,
  setMaskData,
  setProgress,
  appendHistory,
  setCurrentHistoryIndex,
} from '@/lib/features/preview/previewSlice'

interface Img2ImgRequest {
  image: string
  prompt: string
  strength: number
  guidance_scale: number
  negative_prompt: string
  prompt_2: string
  negative_prompt_2: string
}

interface InpaintRequest {
  image: string
  prompt: string
  mask: string
  strength: number
  guidance_scale: number
  negative_prompt: string
  prompt_2: string
  negative_prompt_2: string
  alpha: number
  noise: number
  noise_offset: number
  blur: number
  strict: boolean
  reverse_mask: boolean
}

interface MasksRequest {
  image: string
}

const getPreview = async (
  prompt: string,
  iterations: number,
  guidance_scale: number,
  negative_prompt: string,
  prompt_2: string,
  negative_prompt_2: string
): Promise<string> => {
  console.log('getPreview called with prompt:', prompt)
  const response = await fetch('http://localhost:8000/api/generate', {
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

const img2img = async (request: Img2ImgRequest) => {
  const response = await fetch('http://localhost:8000/api/refine', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    cache: 'no-cache',
    body: JSON.stringify(request),
  })
  const data = await response.json()
  return data.image_data
}

const inpaint = async (request: InpaintRequest) => {
  console.log(`request: ${JSON.stringify(request)}`)
  const response = await fetch('http://localhost:8000/api/inpaint', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    cache: 'no-cache',
    body: JSON.stringify(request),
  })
  const data = await response.json()
  return data.image_data
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

  const generationIterations = useAppSelector((state) => state.controlPanel.generationIterations)
  const generationGuidanceScale = useAppSelector(
    (state) => state.controlPanel.generationGuidanceScale
  )
  const img2imgStrength = useAppSelector((state) => state.controlPanel.img2imgStrength)
  const img2imgGuidanceScale = useAppSelector((state) => state.controlPanel.img2imgGuidanceScale)
  const editorCanvasData = useAppSelector((state) => state.imageEditor.editorCanvasData)
  const previewCanvasData = useAppSelector((state) => state.preview.previewCanvasData)
  const caption = useAppSelector((state) => state.imageEditor.caption)
  const selectedImage = useAppSelector((state) => state.imageEditor.selectedImage)
  const masked = useAppSelector((state) => state.controlPanel.masked)
  const inpaintStrength = useAppSelector((state) => state.controlPanel.inpaintStrength)
  const inpaintGuidanceScale = useAppSelector((state) => state.controlPanel.inpaintGuidanceScale)
  const prompt = useAppSelector((state) => state.promptPanel.prompt)
  const negativePrompt = useAppSelector((state) => state.promptPanel.negativePrompt)
  const prompt2 = useAppSelector((state) => state.promptPanel.prompt2)
  const negativePrompt2 = useAppSelector((state) => state.promptPanel.negativePrompt2)
  const maskData = useAppSelector((state) => state.preview.maskData)
  const maskCount = useAppSelector((state) => state.controlPanel.maskCount)
  const maskIndex = useAppSelector((state) => state.controlPanel.maskIndex)
  const alpha = useAppSelector((state) => state.controlPanel.alpha)
  const maskVisible = useAppSelector((state) => state.controlPanel.maskVisible)
  const disabled = useAppSelector((state) => state.controlPanel.disabled)
  const inpaintNoise = useAppSelector((state) => state.controlPanel.inpaintNoise)
  const inpaintNoiseOffset = useAppSelector((state) => state.controlPanel.inpaintNoiseOffset)
  const inpaintBlur = useAppSelector((state) => state.controlPanel.inpaintBlur)
  const strict = useAppSelector((state) => state.controlPanel.strict)
  const reverseMask = useAppSelector((state) => state.controlPanel.reverseMask)
  const history = useAppSelector((state) => state.preview.history)
  const currentHistoryIndex = useAppSelector((state) => state.preview.currentHistoryIndex)
  const useNoise = useAppSelector((state) => state.controlPanel.useNoise)

  const dispatch = useAppDispatch()

  const [isGenerating, setIsGenerating] = useState(false)
  const [repeats, setRepeats] = useState(1)

  const generateButtonHandler = async () => {
    console.log('Button clicked')
    if (!prompt) return

    const b64 = await getPreview(
      prompt,
      generationIterations,
      generationGuidanceScale,
      negativePrompt,
      prompt2,
      negativePrompt2
    )
    console.log(`b64: ${b64.slice(0, 29)}`)
    dispatch(appendHistory(b64))
    dispatch(setCurrentHistoryIndex(history.length))
    dispatch(toggleDisabled(false))
  }

  const img2imgButtonHandler = async () => {
    console.log('img2img button clicked')
    if (!caption || !history[currentHistoryIndex.value]) return
    let b64 = history[currentHistoryIndex.value] as string
    for (let i = 0; i < repeats; i++) {
      b64 = await img2img({
        image: b64,
        prompt: prompt,
        strength: img2imgStrength,
        guidance_scale: img2imgGuidanceScale,
        negative_prompt: negativePrompt,
        prompt_2: prompt2,
        negative_prompt_2: negativePrompt2,
      })
      dispatch(appendHistory(b64))
      dispatch(setCurrentHistoryIndex(history.length))
      localStorage.setItem('previewCanvasData', b64)
    }
    dispatch(toggleDisabled(false))
  }

  const inpaintWS = () => {
    if (!history[currentHistoryIndex.value]) return
    const ws = new WebSocket('ws://localhost:8000/api/inpaint')
    const request: InpaintRequest = {
      image: history[currentHistoryIndex.value] as string,
      prompt: prompt,
      mask: maskData?.[maskIndex]?.mask,
      strength: inpaintStrength,
      guidance_scale: inpaintGuidanceScale,
      negative_prompt: negativePrompt,
      prompt_2: prompt2,
      negative_prompt_2: negativePrompt2,
      alpha: alpha,
      noise: useNoise ? inpaintNoise : 0,
      noise_offset: useNoise ? inpaintNoiseOffset : 0,
      blur: useNoise ? inpaintBlur : 0,
      strict: strict,
      reverse_mask: reverseMask,
    }

    let i = 0
    let reps = repeats
    dispatch(setProgress(0))
    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      ws.close()
      dispatch(toggleDisabled(false))
    }
    ws.onclose = (event) => {
      console.log('WebSocket closed', event)
    }
    ws.onopen = (event) => {
      console.log('WebSocket connection opened')
      ws.send(JSON.stringify(request))
      ws.onmessage = (message) => {
        console.log(`message keys: `, Object.keys(message))
        const data = JSON.parse(message.data)
        console.log(data)
        if ('image_data' in data) {
          i++
          if (i < reps) {
            dispatch(appendHistory(data.image_data))
            dispatch(setCurrentHistoryIndex(history.length))
            dispatch(setProgress(Math.ceil((i / reps) * 100)))
            request.image = data.image_data
            ws.send(JSON.stringify(request))
          } else if ('step' in data) {
            // Progress message!
            console.log(`Progress: step ${data.step}, t=${data.timestep}`)
          } else {
            console.log(`data keys: ${Object.keys(data)}`)
            dispatch(appendHistory(data.image_data))
            dispatch(setCurrentHistoryIndex(history.length))
            dispatch(setProgress(Math.ceil((i / reps) * 100)))
            dispatch(toggleDisabled(false))
            ws.close()
          }
        } else {
          console.log('WebSocket message received:', data)
        }
      }
    }
  }

  const maskButtonHandler = async () => {
    if (!history[currentHistoryIndex.value]) return
    const ws = new WebSocket('ws://localhost:8000/api/masks')
    const request: MasksRequest = {
      image: history[currentHistoryIndex.value] as string,
    }

    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      ws.close()
      dispatch(toggleDisabled(false))
    }
    ws.onclose = (event) => {
      console.log('WebSocket closed', event)
    }
    ws.onopen = () => {
      dispatch(toggleDisabled(true))
      ws.send(JSON.stringify(request))
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (!Object.hasOwn(data, 'status')) {
          dispatch(setMaskData(data))
          dispatch(toggleDisabled(false))
          ws.close()
        } else {
          console.log('WebSocket message received:', data)
        }
      }
    }
  }

  return (
    <aside className="flex flex-col space-y-2 items-start justify-start w-full p-4">
      <div className="flex flex-col items-start justify-start w-full h-full">
        <h1 className="text-2xl font-bold">Control Panel</h1>
      </div>
      <Accordion type="multiple" collapsible="true" className="w-full h-full space-y-2">
        <AccordionItem value="item-1">
          <AccordionTrigger>Generation</AccordionTrigger>
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
              max={100}
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
              step={0.5}
              value={generationGuidanceScale}
              onChange={(e) => dispatch(setGenerationGuidanceScale(Number(e.target.value)))}
            />
            <Button
              onClick={() => {
                if (disabled) return
                dispatch(toggleDisabled(true))
                generateButtonHandler()
              }}
              disabled={disabled || prompt == ''}
            >
              Generate
            </Button>
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="item-2">
          <AccordionTrigger>Img2Img</AccordionTrigger>
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
              min={0.025}
              max={1}
              step={0.025}
              value={img2imgStrength}
              onChange={(e) => dispatch(setImg2imgStrength(Number(e.target.value)))}
            />
            <label htmlFor="img2imgGuidanceScale">Guidance Scale</label>
            <input
              id="img2imgGuidanceScale"
              type="text"
              value={img2imgGuidanceScale.toString()}
              onChange={(e) => dispatch(setImg2imgGuidanceScale(Number(e.target.value)))}
              className="w-full bg-white text-black p-2 font-bold"
            />
            <input
              type="range"
              min={0}
              max={100}
              step={0.5}
              value={img2imgGuidanceScale}
              onChange={(e) => dispatch(setImg2imgGuidanceScale(Number(e.target.value)))}
            />
            <Button
              onClick={() => {
                if (disabled) return
                dispatch(toggleDisabled(true))
                img2imgButtonHandler()
              }}
              disabled={disabled || prompt == ''}
            >
              Img2Img
            </Button>
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="item-3">
          <AccordionTrigger>Inpaint</AccordionTrigger>
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
              min={0.025}
              max={1}
              step={0.025}
              value={inpaintStrength}
              onChange={(e) => dispatch(setInpaintStrength(Number(e.target.value)))}
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
              step={0.5}
              value={inpaintGuidanceScale}
              onChange={(e) => dispatch(setInpaintGuidanceScale(Number(e.target.value)))}
            />
            <div className="flex items-center gap-3">
              <Checkbox
                id="useNoise"
                checked={useNoise}
                onCheckedChange={(e) => dispatch(toggleUseNoise(!useNoise))}
              />
              <Label htmlFor="useNoise">Use Noise</Label>
            </div>
            <div className="flex flex-col items-center gap-3" hidden={!useNoise}>
              <label htmlFor="inpaintNoise">Noise</label>
              <input
                id="inpaintNoise"
                type="text"
                value={inpaintNoise.toString()}
                onChange={(e) => dispatch(setInpaintNoise(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
                disabled={!useNoise}
              />
              <input
                type="range"
                min={0}
                max={1}
                step={0.025}
                value={inpaintNoise}
                onChange={(e) => dispatch(setInpaintNoise(Number(e.target.value)))}
                disabled={!useNoise}
              />
              <label htmlFor="inpaintNoiseOffset">Noise Offset</label>
              <input
                id="inpaintNoiseOffset"
                type="text"
                value={inpaintNoiseOffset.toString()}
                onChange={(e) => dispatch(setInpaintNoiseOffset(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
                disabled={!useNoise}
              />
              <input
                type="range"
                min={0}
                max={255}
                step={1}
                value={inpaintNoiseOffset}
                onChange={(e) => dispatch(setInpaintNoiseOffset(Number(e.target.value)))}
                disabled={!useNoise}
              />
              <label htmlFor="inpaintBlur">Blur Radius</label>
              <input
                id="inpaintBlur"
                type="text"
                value={inpaintBlur.toString()}
                onChange={(e) => dispatch(setInpaintBlur(Number(e.target.value)))}
                className="w-full bg-white text-black p-2 font-bold"
                disabled={!useNoise}
              />
              <input
                type="range"
                min={0}
                max={10}
                step={1}
                value={inpaintStrength}
                onChange={(e) => dispatch(setInpaintBlur(Number(e.target.value)))}
                disabled={!useNoise}
              />
            </div>
            <label htmlFor="alpha">Alpha: {alpha}</label>
            <input
              type="range"
              min={0}
              max={255}
              step={1}
              value={alpha}
              onChange={(e) => dispatch(setAlpha(Number(e.target.value)))}
            />
            <Button
              onClick={() => {
                // if (disabled) return
                dispatch(toggleDisabled(true))
                inpaintWS()
              }}
              disabled={disabled || !maskData}
            >
              Inpaint
            </Button>
          </AccordionContent>
        </AccordionItem>
        <label htmlFor="repeatsField" className="text-white font-bold">
          Repeats:
          <input
            id="repeatsField"
            type="number"
            min={0}
            max={100}
            step={1}
            value={repeats}
            onChange={(e) => setRepeats(Number(e.target.value))}
          />
        </label>
        <AccordionItem value="item-4">
          <AccordionTrigger>Mask</AccordionTrigger>
          <AccordionContent className="flex flex-col gap-2">
            <div className="flex items-center gap-3">
              <Checkbox
                id="maskVisible"
                checked={maskVisible}
                onCheckedChange={(e) => dispatch(toggleMaskVisible(!maskVisible))}
              />
              <Label htmlFor="maskVisible">Mask Visible</Label>
            </div>
            <div className="flex items-center gap-3">
              <Checkbox
                id="strict"
                checked={strict}
                onCheckedChange={(e) => dispatch(toggleStrict(!strict))}
              />
              <Label htmlFor="strict">Strict</Label>
            </div>
            <div className="flex items-center gap-3">
              <Checkbox
                id="reverseMask"
                checked={reverseMask}
                onCheckedChange={(e) => dispatch(toggleReverseMask(!reverseMask))}
              />
              <Label htmlFor="reverseMask">Reverse Mask</Label>
            </div>
            <label htmlFor="maskIndex">Mask Index: {maskIndex}</label>
            <input
              type="range"
              min={0}
              max={maskData ? maskData.length - 1 : 0}
              value={maskIndex}
              onChange={(e) => dispatch(setMaskIndex(Number(e.target.value)))}
            />
            <Button
              onClick={() => {
                if (disabled) return
                dispatch(setMaskData(null))
                maskButtonHandler()
              }}
              disabled={disabled || !history[currentHistoryIndex.value]}
            >
              Mask
            </Button>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </aside>
  )
}

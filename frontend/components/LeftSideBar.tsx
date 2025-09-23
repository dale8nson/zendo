'use client'

import React, { useState, useEffect, useRef } from 'react'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import { ImageUploadForm } from './ImageUploadForm'
import { ImageGallery } from './ImageGallery'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/Button'
import { Label } from './ui/label'
import { toast } from 'sonner'
import {
  setToken,
  setCaption,
  setSelectedImage,
} from '@/lib/features/image-editor/imageEditorSlice'

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'

import { setCollection, setEditorCanvasStatus } from '@/lib/features/image-editor/imageEditorSlice'
import { toggleDisabled } from '@/lib/features/control-panel/controlPanelSlice'
import { queryOptions, useQuery, useQueryClient } from '@tanstack/react-query'
import { randomUUID } from 'crypto'

export const LeftSideBar = () => {
  const caption = useAppSelector((state) => state.imageEditor.caption)
  const selectedImage = useAppSelector((state) => state.imageEditor.selectedImage)
  // const selectedMaskData = useAppSelector((state) => state.imageEditor.selectedMaskData)
  const collection = useAppSelector((state) => state.controlPanel.collection)
  const scaledSelectionBox = useAppSelector((state) => state.imageEditor.scaledSelectionBox)
  const editorCanvasData = useAppSelector((state) => state.imageEditor.editorCanvasData)
  const editorCanvasStatus = useAppSelector((state) => state.imageEditor.status)
  const objectCaption = useAppSelector((state) => state.imageEditor.objectCaption)
  const maskData = useAppSelector((state) => state.imageEditor.maskData)
  const token = useAppSelector((state) => state.imageEditor.token)

  const dispatch = useAppDispatch()

  const [images, setImages] = useState<string[]>([])
  const [concept, setConcept] = useState<string>('')
  const [trainSteps, setTrainSteps] = useState<number>(100)
  const [repeats, setRepeats] = useState(100)
  const [lr, setLr] = useState<number>(1e-4)
  const [resetOptim, setResetOptim] = useState(false)
  const [resetLrScheduler, setResetLrScheduler] = useState(false)
  const [posterizeBits, setPosterizeBits] = useState(32)
  const [threshold1, setThreshold1] = useState<number>(100)
  const [threshold2, setThreshold2] = useState<number>(200)
  const [apertureSize, setApertureSize] = useState<number>(3)
  const [l2GradientChecked, setL2GradientChecked] = useState<boolean>(false)

  const debounce = useRef(false)

  // Ensure OpenCV Canny-compatible values (3, 5, or 7)
  const normalizeApertureSize = (n: number): number => {
    if (!Number.isFinite(n)) return 3
    if (n <= 4) return 3
    if (n <= 6) return 5
    return 7
  }

  const addToDatasetHandler = async () => {
    if (!selectedImage) return

    // const ws = new WebSocket('ws://10.0.0.22:8002/ws/dataset')
    const ws = new WebSocket('ws://127.0.0.1:8000/ws/dataset')

    const oldStatus = editorCanvasStatus
    const [x, y, w, h] = scaledSelectionBox
    const bbox = [Math.ceil(x), Math.ceil(y), Math.ceil(x + w), Math.ceil(y + h)]
    console.log(`bbox: `, bbox)

    const selectedMaskData = maskData.filter((data) => data.include || data.exclude)

    const message = {
      image_data: selectedImage.image_data,
      bbox,
      collection,
      token,
      caption: caption,
    }

    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      ws.close()
      dispatch(setEditorCanvasStatus(oldStatus))
      dispatch(toggleDisabled(false))
    }
    ws.onclose = (event) => {
      console.log('WebSocket closed', event)
      dispatch(setEditorCanvasStatus(oldStatus))
      dispatch(toggleDisabled(false))
    }
    ws.onopen = () => {
      console.log(`editorCanvasData: ${editorCanvasData?.slice(0, 39)}`)
      dispatch(toggleDisabled(true))
      ws.send(JSON.stringify(message))
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if ('status' in data) {
          toast.success('Image added to dataset')
          dispatch(toggleDisabled(false))
          ws.close()
        } else {
          console.log('WebSocket message received:', data)
        }
      }
    }
  }

  const tokenize = async (text: string) => {
    const response = await fetch('http://127.0.0.1:8000/api/tokenize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    })
    const { tokens } = await response.json()
    return tokens
  }

  const loadEmbeddings = (collection: string) => {
    const ws = new WebSocket('ws://127.0.0.1:8001/ws/load_inversion')

    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      ws.close()
    }
    ws.onclose = (event) => {
      console.log('WebSocket closed', event)
    }

    ws.onopen = () => {
      ws.send(JSON.stringify({ collection, token }))
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data)
        if (message.status === 'OK') {
          toast.success('Embeddings successfully added to pipeline')
        }
      }
    }
  }

  const posterize = async () => {
    if (!selectedImage) return
    const response = await fetch('http://127.0.0.1:8000/api/image/posterize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image_data: selectedImage?.image_data, bits: posterizeBits }),
    })

    const { image_data } = await response.json()

    const meta: MetadataEntry = {
      ...selectedImage,
      id: Number(`0x${crypto.randomUUID().replace('-', '')}`),
      image_data,
    }

    dispatch(setSelectedImage({ ...meta }))

  }

  const trainHandler = () => {
    // let ws = new WebSocket('ws://10.0.0.22:8002/ws/train')
    let ws = new WebSocket('ws://127.0.0.1:8001/ws/train')

    const oldStatus = editorCanvasStatus
    console.log(`collection: ${collection} token: ${token}`)
    const [x, y, w, h] = scaledSelectionBox
    const bbox = [x, y, x + w, y + h].map((n) => Math.ceil(n))
    const message = {
      collection,
      token,
      initializer_token: concept,
      max_train_steps: trainSteps,
      num_training_steps: trainSteps,
      repeats: trainSteps,
      lr,
      reset_optim: resetOptim,
      reset_lr_scheduler: resetLrScheduler,
    }

    ws.onerror = (event) => {
      console.error('WebSocket error', event)
      ws.close()
      dispatch(setEditorCanvasStatus(oldStatus))
      dispatch(toggleDisabled(false))
    }
    ws.onclose = (event) => {
      console.log('WebSocket closed', event)
      dispatch(setEditorCanvasStatus(oldStatus))
      dispatch(toggleDisabled(false))
    }
    ws.onopen = () => {
      console.log(`editorCanvasData: ${editorCanvasData?.slice(0, 39)}`)
      dispatch(toggleDisabled(true))
      console.log(`message: `, message)
      ws.send(JSON.stringify(message))
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if ('status' in data) {
          if (data.status == 'OK') {
            dispatch(toggleDisabled(false))
            ws.close()

            // ws = new WebSocket('ws://127.0.0.1:8001/ws/load_inversion')

            // ws.onerror = (event) => {
            //   console.error('WebSocket error', event)
            //   ws.close()
            //   dispatch(setEditorCanvasStatus(oldStatus))
            //   dispatch(toggleDisabled(false))
            // }
            // ws.onclose = (event) => {
            //   console.log('WebSocket closed', event)
            //   dispatch(setEditorCanvasStatus(oldStatus))
            //   dispatch(toggleDisabled(false))
            // }

            // ws.onopen = (event) => {
            //   console.log('ws.onopen:', event)
            //   ws.send(JSON.stringify({ collection, token }))
            //   ws.onmessage = (event) => {
            //     console.log('ws.onmessage')
            //     const message = JSON.parse(event.data)
            //     if (message.status === 'OK') {
            //       toast.success('Embeddings successfully added to pipeline')
            //     }
            //   }
            // }
          } else {
            console.log('WebSocket message received:', data)
          }
        }
      }
    }
  }

  const encodeDecode = async () => {
    if (!selectedImage) return
    console.log(`selectedImage.image_data: ${selectedImage.image_data.slice(0, 100)}...`)
    const res = await fetch('http://127.0.0.1:8000/api/image/posterize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image_data: selectedImage.image_data }),
    })

    if (!res.ok) {
      alert(`${res.status}: ${res.statusText}`)
    const { image_data } = await res.json()

    console.log('image_data: ', image_data)

    // ensure required MetadataEntry string fields are present
    const meta: MetadataEntry = {
      ...selectedImage,
      id: Number(`0x${crypto.randomUUID().replace('-', '')}`),
      image_data,
      filename: selectedImage?.filename ?? '',
      original_filename: selectedImage?.original_filename ?? '',
    }

    dispatch(setSelectedImage({ ...meta }))
    }

    dispatch(setSelectedImage({ ...meta }))
  }

  useEffect(() => {
    // loadEmbeddings(collection, token)
    console.log(`collection changed to ${collection}`)
  }, [collection])


  function detectEdges(event?: React.MouseEvent<HTMLButtonElement>): void {
    try {
      event?.preventDefault?.()
    } catch (e) {
      // ignore if event is not provided or doesn't support preventDefault
    }

    if (!selectedImage) {
      toast.error('No image selected')
      return
    }

    // run async work in an IIFE so the function can remain non-async
    ;(async () => {
      dispatch(toggleDisabled(true))
      try {
        const res = await fetch('http://127.0.0.1:8000/api/image/edges', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image_data: selectedImage.image_data,
            threshold1,
            threshold2,
            aperture_size: normalizeApertureSize(apertureSize),
            l2_gradient: l2GradientChecked
          }),
        })

        if (!res.ok) {
          const text = await res.text().catch(() => '')
          console.error('Edge detection failed', res.status, res.statusText, text)
          toast.error(`Edge detection failed: ${res.status}`)
          return
        }

        const { image_data } = await res.json()

        const meta: MetadataEntry = {
          ...selectedImage,
          id: Number(`0x${crypto.randomUUID().replace(/-/g, '')}`),
          image_data,
          filename: selectedImage?.filename ?? '',
          original_filename: selectedImage?.original_filename ?? '',
        }

        dispatch(setSelectedImage({ ...meta }))
        toast.success('Edge detection complete')
      } catch (err) {
        console.error('detectEdges error', err)
        toast.error('Edge detection error')
      } finally {
        dispatch(toggleDisabled(false))
      }
    })()
  }

  return (
    <aside className="relative w-full h-full min-w-[200px] bg-neutral-900 border-r border-neutral-800 p-2 overflow-y-scroll">
      <Accordion type="multiple" className="w-full h-full">
        <AccordionItem value="upload">
          <AccordionTrigger position="right">Image Upload</AccordionTrigger>
          <AccordionContent>
            <ImageUploadForm />
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="gallery">
          <AccordionTrigger position="right">Gallery</AccordionTrigger>
          <AccordionContent className="h-full">
            <ImageGallery />
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="image-ops">
          <AccordionTrigger position="right">Image</AccordionTrigger>
          <AccordionContent className='space-y-4'>
            <div className="flex flex-col space-y-4">
              <label htmlFor='threshold1'>Threshold 1</label>
              <input id='threshold1' type='text' className='text-black bg-white' placeholder={threshold1.toString()} onChange={(e) => setThreshold1(Number(e.target.value))} />
              <input type='range' value={threshold1} min={0} max={1000} step={1} onChange={e => setThreshold1(Number(e.target.value))} />
              <label htmlFor='threshold2'>Threshold 2</label>
              <input id='threshold2' type='text' className='text-black bg-white' placeholder={threshold2.toString()} onChange={(e) => setThreshold2(Number(e.target.value))} />
               <input type='range' value={threshold2} min={0} max={1000} step={1} onChange={e => setThreshold2(Number(e.target.value))} />
               <label htmlFor='apertureSize'>Aperture Size</label>
              <input id='apertureSize' type='text' className='text-black bg-white ' placeholder={apertureSize.toString()} onChange={(e) => setApertureSize(normalizeApertureSize(Number(e.target.value)))} />
               <input type='range' value={apertureSize} min={3} max={7} step={2} onChange={e => setApertureSize(normalizeApertureSize(Number(e.target.value)))} />
               <div className='flex w-full'>
                <input id='l2Gradient' type='checkbox' checked={l2GradientChecked} onChange={e => setL2GradientChecked(e.target.checked)} />
                <label htmlFor='l2Gradient' className='text-white'>Use L2 Gradient</label>
               </div>
              <Button onClick={detectEdges}>Detect Edges</Button>
              </div>
            <div className="flex flex-col space-y-4">
              <label htmlFor='bits'>Bits</label>
              <input id='bits' type='text' className='bg-white text-black' placeholder={posterizeBits.toString()} onChange={(e) => setPosterizeBits(Number(e.target.value))} />
              <Button onClick={posterize}>Posterize</Button>
              </div>
            <Button onClick={encodeDecode}>Encode-Decode</Button>
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="data">
          <AccordionTrigger position="right">Data</AccordionTrigger>
          <AccordionContent>
            <Input
              id="concept"
              type="text"
              placeholder={concept}
              className="valid:border-green-400 invalid:border-red-500"
              onChange={(e) => {
                e.target.setCustomValidity('Token does not exist in pretrained model')
              }}
              onKeyDown={async (e) => {
                if (!e.key == 'Enter') return
                const tokens = await tokenize(e.target.value)
                if (tokens.length != 1) {
                  e.target.setCustomValidity('Token does not exist in pretrained model')
                  e.target.checkValidity()
                } else {
                  e.target.setCustomValidity('')
                  e.target.checkValidity()
                  setConcept(e.target.value)
                }
              }}
            />
            <Input
              id="token"
              type="text"
              placeholder={token}
              className="valid:border-green-400 invalid:border-red-500"
              onChange={(e) => {
                dispatch(setCaption(caption?.replaceAll(e.target.value, `<${e.target.value}>`)))
              }}
              onKeyDown={async (e) => {
                if (!e.key == 'Enter') return
                const tokens = await tokenize(e.target.value)
                if (tokens.length === 1) {
                  e.target.setCustomValidity('Token already exists')
                } else {
                  dispatch(setToken(e.target.value))
                  let cap: string = caption as string
                  cap = cap?.replaceAll(/<(.+?)>/g, '$1')
                  cap = cap?.replaceAll(e.target.value, `<${e.target.value}>`)
                  dispatch(setCaption(cap))
                  e.target.setCustomValidity('')
                }
                e.target.checkValidity()
              }}
            />
            <Button onClick={addToDatasetHandler}>Add to Dataset</Button>
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="train">
          <AccordionTrigger position="right">Train</AccordionTrigger>
          <AccordionContent>
            <Input
              id="train-steps"
              type="text"
              defaultValue={100}
              onChange={(e) => {
                setTrainSteps(Number(e.target.value))
              }}
            />
            <Input
              id="lr"
              type="text"
              defaultValue={'1e-4'}
              onChange={(e) => {
                setLr(Number(e.target.value))
              }}
            />
            <div className="flex items-center space-x-2">
              <input
                id="resetLrScheduler"
                type="checkbox"
                checked={resetLrScheduler}
                onChange={(e) => setResetLrScheduler(e.target.checked)}
              />
              <label htmlFor="resetOptim">Reset LR Scheduler</label>
              <input
                id="resetOptim"
                type="checkbox"
                checked={resetOptim}
                onChange={(e) => setResetOptim(e.target.checked)}
              />
              <label htmlFor="resetOptim">Reset Optimizer</label>
            </div>
            <Button onClick={() => trainHandler()}>Train</Button>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </aside>
  )
}

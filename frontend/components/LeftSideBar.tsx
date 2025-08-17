'use client'

import { useState, useEffect, useRef } from 'react'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import { ImageUploadForm } from './ImageUploadForm'
import { ImageGallery } from './ImageGallery'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/Button'
import { Label } from './ui/label'
import { toast } from 'sonner'
import { setToken } from '@/lib/features/image-editor/imageEditorSlice'

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'

import { setCollection, setEditorCanvasStatus } from '@/lib/features/image-editor/imageEditorSlice'
import { toggleDisabled } from '@/lib/features/control-panel/controlPanelSlice'

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

  const debounce = useRef(false)

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
      image_data: selectedImage?.image_data,
      bbox: bbox,
      prompt: caption,
      initializer_token: concept,
      max_train_steps: trainSteps,
      num_training_steps: trainSteps,
      repeats: trainSteps,
      lr,
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

  useEffect(() => {
    // loadEmbeddings(collection, token)
    console.log(`collection changed to ${collection}`)
  }, [collection])

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
        <AccordionItem value="data">
          <AccordionTrigger position="right">Data</AccordionTrigger>
          <AccordionContent>
            <Input
              id="concept"
              type="text"
              placeholder={concept}
              className="valid:border-green-400 invalid:border-red-500"
              onChange={async (e) => {
                setConcept(e.target.value)
                if (debounce.current) return
                debounce.current = true
                const tokens = await tokenize(e.target.value)
                if (tokens.length != 1) {
                  e.target.setCustomValidity('Token does not exist in pretrained model')
                  e.target.checkValidity()
                } else {
                  e.target.setCustomValidity('')
                  e.target.checkValidity()
                }
                setTimeout(() => (debounce.current = false), 500)
              }}
            />
            <Input
              id="token"
              type="text"
              placeholder={token}
              className="valid:border-green-400 invalid:border-red-500"
              onChange={async (e) => {
                dispatch(setToken(e.target.value))
                // if (!debounce.current) return
                // debounce.current = false
                const tokens = await tokenize(e.target.value)
                if (tokens.length === 1) {
                  e.target.setCustomValidity('Token already exists')
                } else {
                  e.target.setCustomValidity('')
                }
                e.target.checkValidity()
                setTimeout(() => (debounce.current = true), 100)
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
            <Button onClick={() => trainHandler()}>Train</Button>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </aside>
  )
}

'use client'

import { useState } from 'react'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import { ImageUploadForm } from './ImageUploadForm'
import { ImageGallery } from './ImageGallery'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/Button'
import { toast } from 'sonner'

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'

import { setEditorCanvasStatus } from '@/lib/features/image-editor/imageEditorSlice'
import { toggleDisabled } from '@/lib/features/control-panel/controlPanelSlice'

export const LeftSideBar = () => {
  const caption = useAppSelector((state) => state.imageEditor.caption)
  const selectedImage = useAppSelector((state) => state.imageEditor.selectedImage)
  // const selectedMaskData = useAppSelector((state) => state.imageEditor.selectedMaskData)
  const collection = useAppSelector((state) => state.imageEditor.collection)
  const scaledSelectionBox = useAppSelector((state) => state.imageEditor.scaledSelectionBox)
  const editorCanvasData = useAppSelector((state) => state.imageEditor.editorCanvasData)
  const editorCanvasStatus = useAppSelector((state) => state.imageEditor.status)
  const objectCaption = useAppSelector((state) => state.imageEditor.objectCaption)
  const maskData = useAppSelector((state) => state.imageEditor.maskData)

  const dispatch = useAppDispatch()

  const [images, setImages] = useState<string[]>([])
  const [token, setToken] = useState('')

  const addToDatasetHandler = async () => {
    if (!selectedImage) return

    const ws = new WebSocket('ws://127.0.0.1:8000/ws/dataset')
    const oldStatus = editorCanvasStatus
    const [x, y, w, h] = scaledSelectionBox
    const bbox = [Math.floor(x), Math.floor(y), Math.floor(x + w), Math.floor(y + h)]
    console.log(`bbox: `, bbox)

    const selectedMaskData = maskData.filter((data) => data.include || data.exclude)

    const message = {
      image_data: selectedImage.image_data,
      masks: selectedMaskData,
      collection: collection,
      token: token,
      caption: caption,
      object_caption: objectCaption,
      bbox: bbox,
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

  const trainHandler = () => {
    if (!selectedImage) return
    const ws = new WebSocket('ws://127.0.0.1:8001/ws/train')
    const oldStatus = editorCanvasStatus
    console.log(`collection: ${collection} token: ${token}`)
    const message = {
      collection: collection,
      token: token,
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
          if (data.status == 'OK') {
            toast.success('Embeddings successfully added to pipeline')
            dispatch(toggleDisabled(false))
            ws.close()
          }
        } else {
          console.log('WebSocket message received:', data)
        }
      }
    }
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
        <AccordionItem value="data">
          <AccordionTrigger position="right">Data</AccordionTrigger>
          <AccordionContent>
            <Input type="text" placeholder="<token>" onChange={(e) => setToken(e.target.value)} />
            <Button onClick={addToDatasetHandler}>Add to Dataset</Button>
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="train">
          <AccordionTrigger position="right">Train</AccordionTrigger>
          <AccordionContent>
            <Button onClick={trainHandler}>Train</Button>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </aside>
  )
}

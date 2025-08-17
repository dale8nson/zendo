'use client'

import { useState, useEffect, useRef } from 'react'
import { useAppDispatch, useAppSelector } from '@/lib/hooks'
import {
  setPreviewCanvasData,
  appendHistory,
  setCurrentHistoryIndex,
  newImage,
} from '@/lib/features/preview/previewSlice'
import { setSelectedMaskData, setMaskData } from '@/lib/features/image-editor/imageEditorSlice'
import { Toggle } from '@/components/ui/toggle'
import { Button } from '@/components/ui/button'

export function ImageControlPanel() {
  console.log('ImageControlPanel')
  const editorCanvasData = useAppSelector((state) => state.imageEditor.editorCanvasData)
  const previewCanvasData = useAppSelector((state) => state.preview.previewCanvasData)
  const currentHistoryIndex = useAppSelector((state) => state.preview.currentHistoryIndex)
  const history = useAppSelector((state) => state.preview.history)
  const selectedImage = useAppSelector((state) => state.imageEditor.selectedImage)
  const selectedMaskData = useAppSelector((state) => state.imageEditor.selectedMaskData)
  const maskData = useAppSelector((state) => state.imageEditor.maskData)
  const maskIndex = useAppSelector((state) => state.imageEditor.maskIndex)
  const scaledSelectionBox = useAppSelector((state) => state.imageEditor.scaledSelectionBox)
  const maskBox = useAppSelector((state) => state.imageEditor.maskBox)
  const previewCanvasSize = useAppSelector((state) => state.preview.previewCanvasSize)

  const dispatch = useAppDispatch()

  const [buttonPressed, setButtonPressed] = useState(false)

  const ref = useRef<HTMLButtonElement>(null)

  const removeSelect = () => {
    dispatch(
      setSelectedMaskData(
        selectedMaskData.filter(
          (data) =>
            data.point_coords[0] !== maskData[maskIndex.value].point_coords[0] &&
            data.point_coords[1] !== maskData[maskIndex.value].point_coords[1] &&
            data.area !== maskData[maskIndex.value].area
        )
      )
    )
  }

  useEffect(() => {
    if (!ref.current || !selectedImage) return
    const classList = ref.current.classList
    console.log('Button clicked')
    if (buttonPressed) {
      classList.replace('hover:bg-neutral-700', 'hover:bg-neutral-800')
      classList.replace('hover:text-neutral-600', 'hover:text-neutral-400')
      classList.replace('hover:border-neutral-800', 'hover:border-neutral-700')
    } else {
      classList.replace('hover:bg-neutral-800', 'hover:bg-neutral-700')
      classList.replace('hover:text-neutral-400', 'hover:text-neutral-600')
      classList.replace('hover:border-neutral-700', 'hover:border-neutral-600')

      console.log(
        `editorCanvasData: ${editorCanvasData?.slice(0, 29)} history.length: ${history.length} history[${currentHistoryIndex}] ${history[currentHistoryIndex]?.slice(0, 29)}`
      )

      // if (currentHistoryIndex >= 0) {
      //   dispatch(appendHistory(selectedImage.image_data))
      //   dispatch(setCurrentHistoryIndex(history.length))
      // }
    }
  }, [buttonPressed])

  return (
    <div className="h-full flex flex-col items-center justify-center space-y-2 max-w-full bg-neutral-900 p-2">
      <Button
        ref={ref}
        className="cursor-pointer"
        // className="bg-neutral-800 min-w-[4.5rem]] max-h-[4rem] p-2 text-neutral-400 border-neutral-700 flex justify-center items-center text-lg rounded hover:bg-neutral-700 hover:text-neutral-600 hover:border-neutral-800"
        onClick={() => {
          console.log('Button clicked')
          if (!selectedImage) return
          const { width, height, image_data } = selectedImage
          const size = Math.max(width, height)
          const [mx, my] = [(size - width) / 2, (size - height) / 2]
          const [x1, y1] = [mx, my]
          const [x2, y2] = [x1 + width, y1 + height]
          const bbox = [x1, y1, x2, y2]
          // const bbox = [0, 0, selectedImage.width, selectedImage.height]

          dispatch(newImage({ bbox, imageData: image_data }))
          // dispatch(setCurrentHistoryIndex(history.length))
        }}
        // onMouseDown={() => setButtonPressed(true)}
        // onMouseUp={() => setButtonPressed(false)}
        // onMouseEnter={onMouseEnter}
        // onMouseLeave={onMouseLeave}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="48px"
          viewBox="0 -960 960 960"
          width="48px"
          fill="#e8eaed"
        >
          <path d="M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v80h-80v-80H200v560h560v-80h80v80q0 33-23.5 56.5T760-120H200Zm480-160-56-56 103-104H360v-80h367L624-624l56-56 200 200-200 200Z" />
        </svg>
      </Button>
      {/*<Button
        size="lg"
        aria-label="merge select"
        onClick={() => {
          const selectedMask = { ...maskData[maskIndex.value] }
          const [x, y, w, h] = scaledSelectionBox
          selectedMask.bbox = [x, y, x + w, y + h]
          selectedMask.canvas_box = maskBox
          dispatch(setSelectedMaskData([...selectedMaskData, selectedMask]))
          dispatch(
            setMaskData([
              ...maskData.map((data, index) => {
                if (index === maskIndex.value) {
                  data.include == true
                }
                return data
              }),
            ])
          )
        }}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="48px"
          viewBox="0 -960 960 960"
          width="48px"
          fill="#e8eaed"
        >
          <path d="M400-80q-33 0-56.5-23.5T320-160v-160H160q-33 0-56.5-23.5T80-400v-400q0-33 23.5-56.5T160-880h400q33 0 56.5 23.5T640-800v160h160q33 0 56.5 23.5T880-560v400q0 33-23.5 56.5T800-80H400Zm0-80h400v-400H560v-240H160v400h240v240Zm80-320Z" />
        </svg>
      </Button>
      <Button size="lg" aria-label="remove select" onClick={removeSelect}>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="48px"
          viewBox="0 -960 960 960"
          width="48px"
          fill="#e8eaed"
        >
          <path d="m500-120-56-56 142-142-142-142 56-56 142 142 142-142 56 56-142 142 142 142-56 56-142-142-142 142Zm-220 0v-80h80v80h-80Zm-80-640h-80q0-33 23.5-56.5T200-840v80Zm80 0v-80h80v80h-80Zm160 0v-80h80v80h-80Zm160 0v-80h80v80h-80Zm160 0v-80q33 0 56.5 23.5T840-760h-80ZM200-200v80q-33 0-56.5-23.5T120-200h80Zm-80-80v-80h80v80h-80Zm0-160v-80h80v80h-80Zm0-160v-80h80v80h-80Zm640 0v-80h80v80h-80Z" />
        </svg>
      </Button>*/}
    </div>
  )
}

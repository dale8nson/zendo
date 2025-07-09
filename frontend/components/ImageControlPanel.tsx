'use client'

import { useState, useEffect, useRef } from 'react'
import { useAppDispatch, useAppSelector } from '@/lib/hooks'
import {
  setPreviewCanvasData,
  appendHistory,
  setCurrentHistoryIndex,
} from '@/lib/features/preview/previewSlice'

export function ImageControlPanel() {
  console.log('ImageControlPanel')
  const editorCanvasData = useAppSelector((state) => state.imageEditor.editorCanvasData)
  const previewCanvasData = useAppSelector((state) => state.preview.previewCanvasData)
  const currentHistoryIndex = useAppSelector((state) => state.preview.currentHistoryIndex)
  const history = useAppSelector((state) => state.preview.history)
  const selectedImage = useAppSelector((state) => state.imageEditor.selectedImage)

  const dispatch = useAppDispatch()

  const [buttonPressed, setButtonPressed] = useState(false)

  const ref = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!ref.current || !editorCanvasData) return
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
        `editorCanvasData: ${editorCanvasData?.slice(0, 29)} history.length: ${history.length} history[${currentHistoryIndex.value}] ${history[currentHistoryIndex.value]?.slice(0, 29)}`
      )

      if (
        editorCanvasData != history?.[currentHistoryIndex.value] ||
        currentHistoryIndex.value === 0
      ) {
        dispatch(appendHistory(editorCanvasData))
        dispatch(setCurrentHistoryIndex(history.length))
      }
    }
  }, [buttonPressed])

  return (
    <div className="h-full flex flex-col items-center justify-center space-y-2 max-w-full bg-neutral-900 p-2">
      <button
        ref={ref}
        className="bg-neutral-800 min-w-[4.5rem]] max-h-[4rem] p-2 text-neutral-400 border-neutral-700 flex justify-center items-center text-lg rounded hover:bg-neutral-700 hover:text-neutral-600 hover:border-neutral-800"
        // onClick={() => {
        //   console.log('Button clicked')
        //   dispatch(setPreviewCanvasData(editorCanvasData))
        // }}
        onMouseDown={() => setButtonPressed(true)}
        onMouseUp={() => setButtonPressed(false)}
        // onMouseEnter={onMouseEnter}
        // onMouseLeave={onMouseLeave}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="24px"
          viewBox="0 -960 960 960"
          width="24px"
          fill="#e8eaed"
        >
          <path d="M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v80h-80v-80H200v560h560v-80h80v80q0 33-23.5 56.5T760-120H200Zm480-160-56-56 103-104H360v-80h367L624-624l56-56 200 200-200 200Z" />
        </svg>
      </button>
    </div>
  )
}

import { useEffect } from 'react'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import {
  setPrompt,
  setNegativePrompt,
  setPrompt2,
  setNegativePrompt2,
} from '@/lib/features/prompt-panel/promptPanelSlice'

export function PromptPanel() {
  const prompt = useAppSelector((state) => state.promptPanel.prompt)
  const negativePrompt = useAppSelector((state) => state.promptPanel.negativePrompt)
  const prompt2 = useAppSelector((state) => state.promptPanel.prompt2)
  const negativePrompt2 = useAppSelector((state) => state.promptPanel.negativePrompt2)
  const dispatch = useAppDispatch()

  useEffect(() => {
    dispatch(setPrompt(localStorage.getItem('prompt') || ''))
    dispatch(setNegativePrompt(localStorage.getItem('negativePrompt') || ''))
    dispatch(setPrompt2(localStorage.getItem('prompt2') || ''))
    dispatch(setNegativePrompt2(localStorage.getItem('negativePrompt2') || ''))
  }, [])

  return (
    <div className="grid grid-cols-2 gap-2 space-x-2 justify-start w-full h-full px-2">
      <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
        <legend className="text-white font-bold px-2">Prompt</legend>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => {
            dispatch(setPrompt(e.target.value))
            localStorage.setItem('prompt', e.target.value)
          }}
          className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
        />
      </fieldset>
      <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
        <legend className="text-white font-bold px-2">Negative Prompt</legend>
        <textarea
          id="negative_prompt"
          value={negativePrompt}
          onChange={(e) => {
            dispatch(setNegativePrompt(e.target.value))
            localStorage.setItem('negativePrompt', e.target.value)
          }}
          className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
        />
      </fieldset>

      <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
        <legend className="text-white font-bold px-2">Prompt 2</legend>
        <textarea
          id="prompt_2"
          value={prompt2}
          onChange={(e) => {
            dispatch(setPrompt2(e.target.value))
            localStorage.setItem('prompt2', e.target.value)
          }}
          className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
        />
      </fieldset>
      <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
        <legend className="text-white font-bold px-2">Negative Prompt 2</legend>
        <textarea
          id="negative_prompt_2"
          value={negativePrompt2}
          onChange={(e) => {
            dispatch(setNegativePrompt2(e.target.value))
            localStorage.setItem('negativePrompt2', e.target.value)
          }}
          className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
        />
      </fieldset>
    </div>
  )
}

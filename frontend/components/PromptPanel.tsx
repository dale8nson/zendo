'use client'

import { useEffect } from 'react'
import { useAppSelector, useAppDispatch } from '@/lib/hooks'
import {
  setGeneratePrompt,
  setGeneratePrompt2,
  setGenerateNegativePrompt,
  setGenerateNegativePrompt2,
  setImg2imgPrompt,
  setImg2imgPrompt2,
  setImg2imgNegativePrompt,
  setImg2imgNegativePrompt2,
  setInpaintPrompt,
  setInpaintPrompt2,
  setInpaintNegativePrompt,
  setInpaintNegativePrompt2,
  setDefaultPromptTab,
  setSelectedTab,
} from '@/lib/features/prompt-panel/promptPanelSlice'

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

export function PromptPanel() {
  const generatePrompt = useAppSelector((state) => state.promptPanel.generate_prompt)
  const generateNegativePrompt = useAppSelector(
    (state) => state.promptPanel.generate_negativePrompt
  )
  const generatePrompt2 = useAppSelector((state) => state.promptPanel.generate_prompt2)
  const generateNegativePrompt2 = useAppSelector(
    (state) => state.promptPanel.generate_negativePrompt2
  )
  const img2imgPrompt = useAppSelector((state) => state.promptPanel.img2img_prompt)
  const img2imgNegativePrompt = useAppSelector((state) => state.promptPanel.img2img_negativePrompt)
  const img2imgPrompt2 = useAppSelector((state) => state.promptPanel.img2img_prompt2)
  const img2imgNegativePrompt2 = useAppSelector(
    (state) => state.promptPanel.img2img_negativePrompt2
  )
  const inpaintPrompt = useAppSelector((state) => state.promptPanel.inpaint_prompt)
  const inpaintNegativePrompt = useAppSelector((state) => state.promptPanel.inpaint_negativePrompt)
  const inpaintPrompt2 = useAppSelector((state) => state.promptPanel.inpaint_prompt2)
  const inpaintNegativePrompt2 = useAppSelector(
    (state) => state.promptPanel.inpaint_negativePrompt2
  )
  const defaultPromptTab = useAppSelector((state) => state.promptPanel.default_prompt_tab)
  const selectedTab = useAppSelector((state) => state.promptPanel.selected_tab)
  const dispatch = useAppDispatch()

  useEffect(() => {
    dispatch(setGeneratePrompt(localStorage.getItem('generate_prompt') || ''))
    dispatch(setGenerateNegativePrompt(localStorage.getItem('generate_negative_prompt') || ''))
    dispatch(setGeneratePrompt2(localStorage.getItem('generate_prompt_2') || ''))
    dispatch(setGenerateNegativePrompt2(localStorage.getItem('generate_negative_prompt_2') || ''))
    dispatch(setImg2imgPrompt(localStorage.getItem('img2img_prompt') || ''))
    dispatch(setImg2imgNegativePrompt(localStorage.getItem('img2img_negative_prompt') || ''))
    dispatch(setImg2imgPrompt2(localStorage.getItem('img2img_prompt_2') || ''))
    dispatch(setImg2imgNegativePrompt2(localStorage.getItem('img2img_negative_prompt_2') || ''))
    dispatch(setInpaintPrompt(localStorage.getItem('inpaint_prompt') || ''))
    dispatch(setInpaintNegativePrompt(localStorage.getItem('inpaint_negative_prompt') || ''))
    dispatch(setInpaintPrompt2(localStorage.getItem('inpaint_prompt_2') || ''))
    dispatch(setInpaintNegativePrompt2(localStorage.getItem('inpaint_negative_prompt_2') || ''))
    dispatch(setDefaultPromptTab(localStorage.getItem('default_prompt_tab') || 'generate'))
  }, [])

  return (
    <Tabs className="flex flex-col w-full h-full m-0" defaultValue={defaultPromptTab}>
      <TabsList className="flex w-full items-center space-x-4">
        <TabsTrigger className="font-bold text-lg" value="generate">
          Generate
        </TabsTrigger>
        <TabsTrigger className="font-bold text-lg" value="img2img">
          Img2Img
        </TabsTrigger>
        <TabsTrigger className="font-bold text-lg" value="inpaint">
          Inpaint
        </TabsTrigger>
      </TabsList>
      <TabsContent className="w-full h-full m-0" value="generate">
        <div className="grid grid-cols-2 gap-2 w-full h-full px-2 m-0">
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt</legend>
            <textarea
              id="generate_prompt"
              value={generatePrompt}
              onChange={(e) => {
                dispatch(setGeneratePrompt(e.target.value))
                localStorage.setItem('generate_prompt', e.target.value)
              }}
              className="bg-neutral-800 text-white !w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt</legend>
            <textarea
              id="negative_prompt"
              value={generateNegativePrompt}
              onChange={(e) => {
                dispatch(setGenerateNegativePrompt(e.target.value))
                localStorage.setItem('generate_negative_prompt', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt 2</legend>
            <textarea
              id="generate_prompt_2"
              value={generatePrompt2}
              onChange={(e) => {
                dispatch(setGeneratePrompt2(e.target.value))
                localStorage.setItem('generate_prompt_2', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt 2</legend>
            <textarea
              id="generate_negative_prompt_2"
              value={generateNegativePrompt2}
              onChange={(e) => {
                dispatch(setGenerateNegativePrompt2(e.target.value))
                localStorage.setItem('generate_negative_prompt2', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
        </div>
      </TabsContent>
      <TabsContent value="img2img">
        <div className="grid grid-cols-2 gap-2 space-x-2 justify-start w-full h-full px-2">
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt</legend>
            <textarea
              id="img2img_prompt"
              value={img2imgPrompt}
              onChange={(e) => {
                dispatch(setImg2imgPrompt(e.target.value))
                localStorage.setItem('img2img_prompt', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt</legend>
            <textarea
              id="img2img_negative_prompt"
              value={img2imgNegativePrompt}
              onChange={(e) => {
                dispatch(setImg2imgNegativePrompt(e.target.value))
                localStorage.setItem('img2img_negative_prompt', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>

          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt 2</legend>
            <textarea
              id="img2img_prompt_2"
              value={img2imgPrompt2}
              onChange={(e) => {
                dispatch(setImg2imgPrompt2(e.target.value))
                localStorage.setItem('img2img_prompt_2', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt 2</legend>
            <textarea
              id="img2img_negative_prompt_2"
              value={img2imgNegativePrompt2}
              onChange={(e) => {
                dispatch(setImg2imgNegativePrompt2(e.target.value))
                localStorage.setItem('img2img_negative_prompt_2', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
        </div>
      </TabsContent>
      <TabsContent value="inpaint">
        <div className="grid grid-cols-2 gap-2 space-x-2 justify-start w-full h-full px-2">
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt</legend>
            <textarea
              id="inpaint_prompt"
              value={inpaintPrompt}
              onChange={(e) => {
                dispatch(setInpaintPrompt(e.target.value))
                localStorage.setItem('inpaint_prompt', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt</legend>
            <textarea
              id="inpaint_negative_prompt"
              value={inpaintNegativePrompt}
              onChange={(e) => {
                dispatch(setInpaintNegativePrompt(e.target.value))
                localStorage.setItem('inpaint_negative_prompt', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Prompt 2</legend>
            <textarea
              id="inpaint_prompt_2"
              value={inpaintPrompt2}
              onChange={(e) => {
                dispatch(setInpaintPrompt2(e.target.value))
                localStorage.setItem('inpaint_prompt_2', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
          <fieldset className="border-solid border-2 border-neutral-800 w-full h-full p-1">
            <legend className="text-white font-bold px-2">Negative Prompt 2</legend>
            <textarea
              id="inpaint_negative_prompt_2"
              value={inpaintNegativePrompt2}
              onChange={(e) => {
                dispatch(setInpaintNegativePrompt2(e.target.value))
                localStorage.setItem('inpaint_negative_prompt_2', e.target.value)
              }}
              className="bg-neutral-800 text-white w-full h-full p-2 focus-visible:outline-none focus-visible:bg-neutral-700 resize-none"
            />
          </fieldset>
        </div>
      </TabsContent>
    </Tabs>
  )
}

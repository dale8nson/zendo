import { useRef, useState } from 'react'
import { useAppSelector } from '@/lib/hooks'
import { Button } from '@/components/Button'

export const EditorPanel = () => {
  const captionScore =

  return (
    <div className="flex items-start justify-around w-full h-full bg-neutral-900">
      <textarea
        ref={textRef}
        className="text-lg text-white w-4/5 h-full m-0 px-2 resize-none border-2 border-solid border-neutral-800"
        defaultValue={caption || 'Loading...'}
        onChange={(e) => dispatch(setCaption(e.target.value))}
      />
      <div className="flex flex-col items-center justify-between h-full w-1/5 p-2 border-2 border-solid border-neutral-800">
        <h1 className="text-lg font-bold text-white">
          Match Score:{' '}
          {captionScoreLoading
            ? 'Loadin...'
            : captionScore
              ? `${(captionScore as number).toFixed(2)}%`
              : (0).toFixed(2)}
        </h1>
        <output></output>
        <Button onClick={scoreButtonClickHandler}>Score</Button>
      </div>
    </div>
  )
}

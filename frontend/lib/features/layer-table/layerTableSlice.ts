import { createSlice } from '@reduxjs/toolkit'

interface Layer {
  id: string
  order: number
  active: boolean
  label: string
  visible: boolean
  opacity: number
  image_data: string
}

interface LayerTableState {
  layers: Layer[]
}

const initialState: LayerTableState = {
  layers: [
    {
      id: crypto.randomUUID(),
      order: 0,
      active: true,
      label: 'root',
      visible: true,
      opacity: 1,
      image_data: '',
    },
  ],
}

const layerTableSlice = createSlice({
  name: 'layerTable',
  initialState,
  reducers: {
    setLayers: (state, action) => {
      state.layers = action.payload
    },
  },
})

export default layerTableSlice.reducer
export const { setLayers } = layerTableSlice.actions

import express from 'express'
import { createProxyMiddleware } from 'http-proxy-middleware'

const app = express()
const PORT = process.env.PORT || 8080

app.use(
  '/api',
  createProxyMiddleware({
    target: 'http://localhost:8000',
    changeOrigin: true,
    pathRewrite: {
      '^/api': '/api',
    },
  })
)

app.use('/', {
  target: 'http://localhost:3000',
  changeOrigin: true,
})

app.listen(PORT, () => {
  console.log(`Reverse proxy listening on port ${PORT}`)
})

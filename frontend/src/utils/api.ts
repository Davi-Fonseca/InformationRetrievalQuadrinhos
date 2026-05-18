import axios from "axios"

const baseURL = typeof window != "undefined" ? process.env.NEXT_PUBLIC_API : process.env.NEXT_PUBLIC_DOCKER_API

const api = axios.create({
  baseURL,
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
})

export default api
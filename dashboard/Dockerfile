FROM node:20-alpine

WORKDIR /app/dashboard
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
# The browser (on your machine) calls the API, so this is the HOST's address.
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]

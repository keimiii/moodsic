#!/bin/bash

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing React dependencies..."
    npm install
fi

# Start the React development server
echo "Starting React frontend..."
echo "Frontend will be available at: http://localhost:3000"
npm start

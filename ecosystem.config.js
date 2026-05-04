module.exports = {
  apps: [{
    name:        'echosight',
    script:      'server.js',
    interpreter: 'node',
    watch:       false,
    env: {
      NODE_ENV: 'development',
      PORT:     3000
    }
  }]
}

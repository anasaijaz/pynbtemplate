<template>
  <div class="container">
    <p v-if="execution_count" class="line_in">Out [{{execution_count}}]:</p>
    <p v-else>***</p>
    <div class="block">
    <div class="output" :key="index" v-for="(output, index) in outputs">
      <div v-if="type(output) === 'stream'">
        <pre>{{ !output['text'] ? '' : output['text'].join('') }}</pre>
      </div>

      <div v-if="type(output) === 'execute_result'">
        <pre>{{ !output['data']['text/plain'] ? '' : output['data']['text/plain'].join('') }}</pre>
      </div>

      <div v-if="type(output) === 'display_data' && output['data'] !== undefined">
        <img v-if="haveImage(output['data']) === 'IMAGE'"
             :src="displayData('IMAGE', output).data"
             :width="displayData('IMAGE', output).width"
             :height="displayData('IMAGE', output).height"
        />
        <pre v-if="haveText(output['data']) === 'TEXT'">{{displayData('TEXT', output).data}}</pre>
      </div>

      <div v-if="type(output) === 'error'">
        <pre>{!output.traceback ? undefined : output.traceback.join('\n')}</pre>
      </div>
    </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "CellOutput",
  props: ['outputs', 'execution_count'],
  methods: {
    type(output) {
      return output['output_type']
    },
    haveImage(data){
      if('image/png' in data){
        return 'IMAGE'
      }
    },
    haveText(data){
      if('text/plain' in data){
        return 'TEXT'
      }
    },
    haveHTML(data){
      if('text/plain' in data) {
        return 'HTML'
      }
    },
    displayData(type, output) {
      const data = output['data']
      const metaData = output['metadata']
      if(type==='IMAGE') {
        let size = metaData && metaData['image/png']
        return {
          data: `data:image/png;base64,${data['image/png']}`,
          width: size? size['width'] : 'auto',
          height: size? size['height'] : 'auto'
        }
      }
      if(type==='TEXT') {
        return {
          data: data['text/plain'].join('')
        }
      }
    },
    output() {

    }
  },
  computed: {

  }
}
</script>

<style scoped>
.line_in {
  font-family: Consolas, monospace;
  margin: unset;
  font-size: small;
}

.container {
  display: flex;
  gap: 1.1rem;
}

.block {
  flex-grow: 1;
  overflow: auto;
}
</style>

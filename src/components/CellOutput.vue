<template>
  <div class="container">
    <p v-if="execution_count" class="line_in">Out [{{execution_count}}]:</p>
    <p v-else>***</p>
    <div class="block">
    <div class="output" :key="index" v-for="(output, index) in outputs">
      <div v-if="type(output) === 'stream'">
        <pre :class="output['name']==='stderr'?'stderr_block':'stdout_block'">{{ !output['text'] ? '' : output['text'].join('') }}</pre>
      </div>

      <div v-if="type(output) === 'execute_result'">
        <div v-if="'text/html' in output['data']" v-html="output['data']['text/html'].join('')"></div>
        <pre v-else>{{ !output['data']['text/plain'] ? '' : output['data']['text/plain'].join('') }}</pre>
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
      if('text/html' in data) {
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
  white-space: nowrap;
}

.container {
  display: flex;
  gap: 1.1rem;
  padding-top: 1.5rem;
}

.block {
  flex-grow: 1;
  overflow: auto;
}

.stderr_block {
  background-color: RGB(255, 221, 220);
  width: max-content;
}


::v-deep table {
  margin-bottom: 2em;
  width: 100%;
  border: none;
}
::v-deep th {
  font-weight: bold;
  text-align: left;
  border: none;
  border-bottom: 1px solid black;
  padding-block: 0.25rem;
  color: black;
  font-size: 0.8rem;
}
::v-deep td {
  border: unset;
  text-align: right;
}
::v-deep tr:nth-child(2n) {
  background-color: #f5f5f5;
}
::v-deep caption, th, td {
  padding: 4px 10px 4px 0;
}
::v-deep caption {
  background: #f1f1f1;
  padding: 10px 0;
  margin-bottom: 1em;
}

::v-deep tr,td,th {
  vertical-align: middle;
}

</style>

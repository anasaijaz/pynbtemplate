<template>
  <div :class="rootClass" :style="style">
    <div :key="index" v-for="(cell, index) in cells">
      <div class="block">
      <div v-if="isCellValid">
        <CellSource :code_font_size="code_font_size" :execution_count="cell['execution_count']" :cell="cell"  :src="cell['source']"/>
      </div>
      <div v-if="isCellHaveOutput(cell)">
        <CellOutput :execution_count="cell['execution_count']"  :outputs="cell['outputs']"/>
      </div>
      </div>
      <br/>
    </div>
  </div>
</template>

<script>
import CellSource from "@/components/CellSource";
import CellOutput from "@/components/CellOutput";
export default {
  name: 'PynbTemplate',
  components: {CellOutput, CellSource},
  props: {
    // ['json', 'gist', 'height', 'width']
    json:{
      type:Object
    },
    gist: {
      type: Boolean,
      default: false
    },
    height: {
      type: String,
      default: 'max-content'
    },
    width: {
      type: String,
      default: 'max-content'
    },
    code_font_size: {
      type: String,
      default: '1.75rem'
    }
  },
  methods: {
    isCellValid(cell) {
      return cell['cell_type'] !== undefined
    },
    isCellHaveOutput(cell) {
      return cell['outputs'] !== undefined && cell['outputs'].length > 0
    }
  },
  computed: {
    cells(){
      return this.json['cells']
    },
    rootClass() {
      let root = 'root'
      if(this.gist){
        root += ' '
        root += 'gist'
      }
      return root
    },
    style() {
      return `width: ${this.width}; height: ${this.height};`
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>

.root {
  position: relative;
  overflow: auto;
}

.block {
  padding-bottom: 1.25rem;
}

.root.gist {
  padding: 1rem;
  width: 600px;
  height: 600px;
  margin: auto;
  border: 1px solid lightgray;
  border-radius: 8px;
}


::v-deep h2, ::v-deep h3,::v-deep h4,::v-deep h5,::v-deep h6,::v-deep p {
  margin: unset;
}

</style>

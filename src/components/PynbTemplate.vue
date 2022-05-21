<template>
  <div :class="rootClass" :style="style">
  <div >
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
    <div class="footer"> Anas</div>

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
      default: '400px'
    },
    width: {
      type: String,
      default: '400px'
    },
    code_font_size: {
      type: String,
      default: '14px'
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
}

.block {
  padding-bottom: 1.25rem;
}

.root.gist {
  padding: 1rem;
  width: 600px;
  height: 600px;
  overflow: auto;
  margin: auto;
  border: 1px solid lightgray;
  border-radius: 8px;
}

.footer {
  background-color: red;
  width: 100%;
}
</style>

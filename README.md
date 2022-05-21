# pynbtemplate
A Vue component to render python notebooks

Install the component using

``` npm install pynbtemplate ```

Install the Vue plugin inside your vue app

```Vue.use(PynbtemplateSimple)```

Import css

```import 'pynbtemplate/dist/pynbtemplate.css'```

Now you are good to go


| Prop        | Type           | Description  |
| ------------- |:-------------:| -----------:|
| width      | px rem em... | Width of the container |
| height      | px rem em...      |   Height of the container |
| gist | boolean    |    Provide a github gist like look |
| code_font_size | px rem em ...     |    font size of the code |
| json | object     |    JSON schema of the code  |

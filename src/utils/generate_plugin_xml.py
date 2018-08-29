"""
utils.generate_plugin_xml
-------------------------

Provides tools for generating the XML Paraview plugins.

"""

# See blog for details: https://blog.kitware.com/easy-customization-of-the-paraview-python-programmable-filter-property-panel/

import os
import sys
import inspect
import textwrap


def escapeForXmlAttribute(s):

    # http://www.w3.org/TR/2000/WD-xml-c14n-20000119.html#charescaping
    # In character data and attribute values, the character information items "<" and "&" are represented by "&lt;" and "&amp;" respectively.
    # In attribute values, the double-quote character information item (") is represented by "&quot;".
    # In attribute values, the character information items TAB (#x9), newline (#xA), and carriage-return (#xD) are represented by "&#x9;", "&#xA;", and "&#xD;" respectively.

    s = s.replace('&', '&amp;')  # Must be done first!
    s = s.replace('<', '&lt;')
    s = s.replace('>', '&gt;')
    s = s.replace('"', '&quot;')
    s = s.replace('\r', '&#xD;')
    s = s.replace('\n', '&#xA;')
    s = s.replace('\t', '&#x9;')
    return s


def getScriptPropertiesXml(info, function_name=None):

    if function_name is None:
        function_name = 'RequestData'

    requestData = escapeForXmlAttribute(info[function_name])
    requestInformation = escapeForXmlAttribute(info['RequestInformation'])
    requestUpdateExtent = escapeForXmlAttribute(info['RequestUpdateExtent'])

    if requestData:
        requestData = '''
      <StringVectorProperty
        name="Script"
        command="SetScript"
        number_of_elements="1"
        default_values="%s"
        panel_visibility="advanced">
        <Hints>
         <Widget type="multi_line"/>
       </Hints>
      <Documentation>This property contains the text of a python program that
      the programmable source runs.</Documentation>
      </StringVectorProperty>''' % requestData

    if requestInformation:
        requestInformation = '''
      <StringVectorProperty
        name="InformationScript"
        label="RequestInformation Script"
        command="SetInformationScript"
        number_of_elements="1"
        default_values="%s"
        panel_visibility="advanced">
        <Hints>
          <Widget type="multi_line" />
        </Hints>
        <Documentation>This property is a python script that is executed during
        the RequestInformation pipeline pass. Use this to provide information
        such as WHOLE_EXTENT to the pipeline downstream.</Documentation>
      </StringVectorProperty>''' % requestInformation

    if requestUpdateExtent:
        requestUpdateExtent = '''
      <StringVectorProperty
        name="UpdateExtentScript"
        label="RequestUpdateExtent Script"
        command="SetUpdateExtentScript"
        number_of_elements="1"
        default_values="%s"
        panel_visibility="advanced">
        <Hints>
          <Widget type="multi_line" />
        </Hints>
        <Documentation>This property is a python script that is executed during
        the RequestUpdateExtent pipeline pass. Use this to modify the update
        extent that your filter ask up stream for.</Documentation>
      </StringVectorProperty>''' % requestUpdateExtent

    return '\n'.join([requestData, requestInformation, requestUpdateExtent])


def getPythonPathProperty():
    return '''
      <StringVectorProperty command="SetPythonPath"
                            name="PythonPath"
                            number_of_elements="1"
                            panel_visibility="advanced">
        <Documentation>A semi-colon (;) separated list of directories to add to
        the python library search path.</Documentation>
      </StringVectorProperty>'''


def getFilterPropertyXml(propertyInfo, propertyName):

    propertyValue = propertyInfo[propertyName]
    propertyLabel = propertyName.replace('_', ' ')

    if isinstance(propertyValue, list):
        numberOfElements = len(propertyValue)
        assert numberOfElements > 0
        propertyType = type(propertyValue[0])
        defaultValues = ' '.join([str(v) for v in propertyValue])
    else:
        numberOfElements = 1
        propertyType = type(propertyValue)
        defaultValues = str(propertyValue)

    if propertyType is bool:

        defaultValues = defaultValues.replace('True', '1').replace('False', '0')

        return '''
      <IntVectorProperty
        name="%s"
        label="%s"
        initial_string="%s"
        command="SetParameter"
        animateable="1"
        default_values="%s"
        number_of_elements="%s">
        <BooleanDomain name="bool" />
        <Documentation></Documentation>
      </IntVectorProperty>''' % (propertyName, propertyLabel, propertyName, defaultValues, numberOfElements)

    if propertyType is int:
        return '''
      <IntVectorProperty
        name="%s"
        label="%s"
        initial_string="%s"
        command="SetParameter"
        animateable="1"
        default_values="%s"
        number_of_elements="%s">
        <Documentation></Documentation>
      </IntVectorProperty>''' % (propertyName, propertyLabel, propertyName, defaultValues, numberOfElements)

    if propertyType is float:
        return '''
      <DoubleVectorProperty
        name="%s"
        label="%s"
        initial_string="%s"
        command="SetParameter"
        animateable="1"
        default_values="%s"
        number_of_elements="%s">
        <Documentation></Documentation>
      </DoubleVectorProperty>''' % (propertyName, propertyLabel, propertyName, defaultValues, numberOfElements)

    if propertyType is str:
        return '''
      <StringVectorProperty
        name="%s"
        label="%s"
        initial_string="%s"
        command="SetParameter"
        animateable="1"
        default_values="%s"
        number_of_elements="%s">
        <Documentation></Documentation>
      </StringVectorProperty>''' % (propertyName, propertyLabel, propertyName, defaultValues, numberOfElements)

    raise Exception('Unknown property type: %r' % propertyType)


def getFilterPropertiesXml(info):

    propertyInfo = info['Properties']
    xml = [getFilterPropertyXml(propertyInfo, name) for name in sorted(propertyInfo.keys())]
    return '\n\n'.join(xml)


def getNumberOfInputs(info):
    return info.get('NumberOfInputs', 1)


def getInputPropertyXml(info):

    numberOfInputs = getNumberOfInputs(info)
    if not numberOfInputs:
        return ''

    inputDataType = info.get('InputDataType', 'vtkDataObject')

    inputDataTypeDomain = ''
    if inputDataType:
        inputDataTypeDomain = '''
          <DataTypeDomain name="input_type">
            <DataType value="%s"/>
          </DataTypeDomain>''' % inputDataType

    inputPropertyAttributes = 'command="SetInputConnection"'
    if numberOfInputs > 1:
        inputPropertyAttributes = '''\
            clean_command="RemoveAllInputs"
            command="AddInputConnection"
            multiple_input="1"'''

    inputPropertyXml = '''
      <InputProperty
        name="Input"
        %s>
          <ProxyGroupDomain name="groups">
            <Group name="sources"/>
            <Group name="filters"/>
          </ProxyGroupDomain>
          %s
      </InputProperty>''' % (inputPropertyAttributes, inputDataTypeDomain)

    return inputPropertyXml


def getOutputDataSetTypeXml(info):

    outputDataType = info.get('OutputDataType', '')

    typeMap = {
        '': 8,  # same as input
        'vtkPolyData': 0,
        'vtkStructuredGrid': 2,
        'vtkRectilinearGrid': 3,
        'vtkUnstructuredGrid': 4,
        'vtkImageData': 6,
        'vtkUniformGrid': 10,
        'vtkMultiBlockDataSet': 13,
        'vtkHierarchicalBoxDataSet': 15,
        'vtkTable': 19
    }

    typeValue = typeMap[outputDataType]

    return '''
      <!-- Output data type: "%s" -->
      <IntVectorProperty command="SetOutputDataSetType"
                         default_values="%s"
                         name="OutputDataSetType"
                         number_of_elements="1"
                         panel_visibility="never">
        <Documentation>The value of this property determines the dataset type
        for the output of the programmable filter.</Documentation>
      </IntVectorProperty>''' % (outputDataType or 'Same as input', typeValue)


def getProxyGroup(info):
    if "Group" not in info:
        return 'sources' if getNumberOfInputs(info) == 0 else 'filters'
    else:
        return info["Group"]


def generatePythonFilter(info, function_name=None):

    if function_name is None:
        function_name = 'RequestData'

    proxyName = info['Name']
    proxyLabel = info['Label']
    shortHelp = escapeForXmlAttribute(info['Help'])
    longHelp = escapeForXmlAttribute(info['LongHelp'] if 'LongHelp' in info else info['Help'])
    extraXml = info.get('ExtraXml', '')

    proxyGroup = getProxyGroup(info)
    inputPropertyXml = getInputPropertyXml(info)
    outputDataSetType = getOutputDataSetTypeXml(info)
    scriptProperties = getScriptPropertiesXml(info,
                                              function_name=function_name)
    filterProperties = getFilterPropertiesXml(info)

    outputXml = '''\
<ServerManagerConfiguration>
  <ProxyGroup name="%s">
    <SourceProxy name="%s" class="vtkPythonProgrammableFilter" label="%s">

      <Documentation
        long_help="%s"
        short_help="%s">
      </Documentation>

%s

%s

%s

%s

%s

    </SourceProxy>
 </ProxyGroup>
</ServerManagerConfiguration>
      ''' % (proxyGroup, proxyName, proxyLabel, longHelp, shortHelp, inputPropertyXml,
             filterProperties, extraXml, outputDataSetType, scriptProperties)

    return textwrap.dedent(outputXml)


def replaceFunctionWithSourceString(namespace, functionName, allowEmpty=False):

    func = namespace.get(functionName)
    if not func:
        if allowEmpty:
            namespace[functionName] = ''
            return
        else:
            raise Exception('Function %s not found in input source code.' % functionName)

    if not inspect.isfunction(func):
        raise Exception('Object %s is not a function object.' % functionName)

    lines = inspect.getsourcelines(func)[0]

    if len(lines) <= 1:
        raise Exception('Function %s must not be a single line of code.' % functionName)

    # skip first line (the declaration) and then dedent the source code
    sourceCode = textwrap.dedent(''.join(lines[1:]))

    namespace[functionName] = sourceCode


def generatePythonFilterFromFiles(script_file, output_file, function_name=None):
    """Generates an XML plugin file for Paraview from Python source file

    :param function_name: the name of the function containing the plugin entry point.
      Defaults to `RequestData`
    """

    if function_name is None:
        function_name = 'RequestData'

    namespace = {}
    # exec(script_file_content, namespace) exec does not let access the source code w. inspect
    execfile(script_file, namespace)

    replaceFunctionWithSourceString(namespace, function_name)
    replaceFunctionWithSourceString(namespace, 'RequestInformation', allowEmpty=True)
    replaceFunctionWithSourceString(namespace, 'RequestUpdateExtent', allowEmpty=True)

    xmlOutput = generatePythonFilter(namespace, function_name=function_name)

    open(output_file, 'w').write(xmlOutput)


def generate_plugin_after_install(output_file):
    """Utility for generating the plugin XML file after the package has been successfully installed

    This utility uses the `plugin.py` file as an entry point. This is suitable for
    running in Paraview in an environment that is not a virtual environment. It requires a Paraview
    installation of the plugin using `pvpython`. See the documentation for more details.
    """

    import mpi_is_sw.brain_connectivity.plugin as plugin_main

    file_to_process = inspect.getsourcefile(plugin_main)
    generatePythonFilterFromFiles(script_file=file_to_process, output_file=output_file)


def generate_plugin_in_virtualenv_after_install(output_file):
    """Utility for generating the plugin XML file after the package has been successfully installed

    This utility uses the `plugin.py` file as an entry point. This is suitable for
    running in Paraview in an environment that is not a virtual environment. It requires a Paraview
    installation of the plugin using `pvpython`. See the documentation for more details.
    """
    if 'VIRTUAL_ENV' not in os.environ:
        raise RuntimeError("This command should run from a virtual environment.")

    virtual_env_absolute_path = os.path.abspath(os.environ['VIRTUAL_ENV'])
    virtual_env_activation_script = os.path.join(virtual_env_absolute_path, 'bin', 'activate_this.py')

    if not os.path.exists(virtual_env_activation_script):
        raise RuntimeError("Cannot find the activation script in the virtual environment '%s'" % virtual_env_absolute_path)

    import mpi_is_sw.brain_connectivity.plugin as plugin_main

    file_to_process = inspect.getsourcefile(plugin_main)
    file_content = open(file_to_process, 'r').read()

    # substitutes the VIRTUAL_ENV placeholder
    from string import Template

    template = Template(file_content)
    transformed_content = template.substitute(VIRTUAL_ENV_SUBSTITUTION=virtual_env_activation_script)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.py') as f_temp:

        f_temp.write(transformed_content)
        f_temp.flush()
        f_temp.seek(0)

        generatePythonFilterFromFiles(script_file=f_temp.name,
                                      output_file=output_file,
                                      function_name='RequestDataVirtualEnv')


def main_virtualenv():
    if len(sys.argv) != 2:
        print 'Usage: %s <xml output filename>' % sys.argv[0]
        sys.exit(1)

    generate_plugin_in_virtualenv_after_install(output_file=sys.argv[1])


def main():

    if len(sys.argv) != 2:
        print 'Usage: %s <xml output filename>' % sys.argv[0]
        sys.exit(1)

    generate_plugin_after_install(outputFile=sys.argv[1])


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print 'Usage: %s <input_file> <xml output filename>' % sys.argv[0]
        sys.exit(1)

    generatePythonFilterFromFiles(sys.argv[1], sys.argv[2])

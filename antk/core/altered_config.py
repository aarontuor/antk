from __future__ import print_function
import re
from antk.core.node_ops import *
from antk.lib import termcolor
import sys
import os
import traceback

NODE_GLOBALS = globals().copy()

def ph_rep(ph):
    """
    Convenience function for representing a tensorflow placeholder.

    :param ph: A `tensorflow`_ `placeholder`_.
    :return: A string representing the placeholder.
    """
    return 'Placeholder("%s", shape=%s, dtype=%r)' % (ph.name, ph.get_shape().as_list(), ph.dtype)

class UndefinedVariableError(Exception):
    '''Raised when a a variable in config is not a key in variable_bindings map handed to graph_setup.'''
    pass

class UnsupportedNodeError(NameError):
    '''Raised when a config file calls a function that is not defined, i.e., has not been imported, or is not in the
    node_ops base file.'''
    pass

class RandomNodeFunctionError(KeyError):
    '''Raised when something strange happened with a node function call.'''
    pass

class MissingTensorError(Exception):
    '''Raised when a tensor is described by name only in the graph and it is not in a dictionary.'''
    pass

class MissingDataError(Exception):
    '''Raised when data needed to determine shapes is not found in the :any:`DataSet`.'''
    pass

class ProcessLookupError(Exception):
    '''Raised when lookup receives a dataname argument without a corresponding value in it's :any:`DataSet`
    and there is not already a Placeholder with that name.'''
    pass

class GraphMarkerError(Exception):
    '''Raised when leading character of a line (other than first)
    in a graph config file is not the specified level marker.'''
    pass

class AntGraph(object):
    """
    Object to store graph information from graph built with config file.

    :param config: A plain text config file
    :param tensordict: A dictionary of premade tensors represented in the config by key
    :param placeholderdict: A dictionary of premade placeholder tensors represented in the config by key
    :param data: A dictionary of data matrices with keys corresponding to placeholder names in graph.
    :param function_map: A dictionary of function_handle:node_op pairs to use in building the graph
    :param imports: A dictionary of module_name:path_to_module key value pairs for custom node_ops modules.
    :param marker: The marker for representing graph structure
    :param variable_bindings: A dictionary with entries of the form  *variable_name:value* for variable replacement in config file.
    :param graph_name: The name of the graph. Will be used to name the graph pdf file.
    :param graph_dest: The folder to write the graph pdf and graph dot string to.
    :param develop: True|False. Whether to print tensor info, while constructing the tensorflow graph.
    """
    def __init__(self, config, tensordict={}, placeholderdict={}, data=None, function_map={},
                imports={}, marker='-', variable_bindings=None, graph_name='no_name',
                 graph_dest='antpics/', develop=False):

        self.marker = marker
        if data and type(data) is not dict:
            raise TypeError('Data argument to AntGraph constructor must be a python dictionary with keys, corresponding'
                            'to placeholder names, and values of numpy arrays, scipy sparse csr_matrices, or HotIndex objects.')
        self.data = data
        self._tensordict = tensordict
        self._placeholderdict = placeholderdict
        self.develop = develop
        #==========================================================================
        #==========================Node Function Extensions========================
        #==========================================================================
        NODE_GLOBALS.update(function_map)
        self._import_node_files(imports)

        #===============================================================================
        #==================Make Graph===================================================
        #===============================================================================
        with open(config, 'r') as config_file:
            graph_spec = config_file.read().strip()  # remove whitespace at end of file
        graph_spec = self._substitute_variables(graph_spec, variable_bindings).split('\n')
        outputs, node_names = self._get_edges(graph_spec)
        self._dotstring = 'digraph ' + graph_name + ' {'
        self._add_nodes(node_names)
        output_list = []
        for subgraph in outputs:
            output_list.append(self._traverse_graph(subgraph))
        if len(output_list) == 1:
            self._tensor_out = output_list[0]
        else:
            self._tensor_out = output_list

        #===============================================================================
        #==================Make Graphviz Dot Picture====================================
        #===============================================================================
        self._dotstring += '\n}'
        if not graph_dest.endswith('/'):
            graph_dest += '/'
        self._path_to_graph_pic = graph_dest + graph_name + '.pdf'
        os.system('mkdir ' + graph_dest)
        with open(graph_dest + graph_name + '.dot', 'w') as dot_file:
            dot_file.write(self._dotstring)
        os.system('dot -Tpdf -o ' + graph_dest + graph_name + '.pdf ' + graph_dest + graph_name + '.dot')

    #===============================================================================
    #==================PROPERTIES===================================================
    #===============================================================================
    @property
    def tensordict(self):
        '''
        A dictionary of tensors which are nodes in the graph.
        '''
        return self._tensordict

    @property
    def placeholderdict(self):
        '''
        A dictionary of tensors which are placeholders in the graph. The key should correspond to the key of
        the corresponding data in a data dictionary.
        '''
        return self._placeholderdict

    @property
    def tensor_out(self):
        '''
        Tensor or list of tensors returned from last node of graph.
        '''
        return self._tensor_out

    #===============================================================================
    #==================INSTANCE METHODS=============================================
    #===============================================================================
    def display_graph(self, pdfviewer='okular'):
        """
        Display the pdf image of graph from config file to screen.
        """
        os.system(pdfviewer + ' ' + self._path_to_graph_pic + ' &')

    def get_array(collection_name, index, session, graph):
        #return(graph.get_tensor_by_name)
        return session.run(tf.get_collection(collection_name)[index])



    #===============================================================================
    #==================PRIVATE METHODS==============================================
    #===============================================================================
    def _traverse_graph(self, graph):
        """
        This is a postorder 'tree' traversal with possibly repeated non-looping nodes.
        """
        if len(graph) == 1:
            outspec = graph[0]
            return self._make_tensor(outspec)
        else:
            vertex_name = graph[0].strip().split()[0].strip(self.marker[0])
            t_list = []
            edges, node_names = self._get_edges(graph[1:len(graph)])
            self._add_edges(vertex_name, node_names)
            for end_node in edges:
                t_list.append(self._traverse_graph(end_node))
            spec = graph[0]
            if len(edges) == 1:
                tensor_out = self._make_tensor(spec, intensors=t_list[0])
            else:
                tensor_out = self._make_tensor(spec, intensors=t_list)
            return tensor_out

    def _make_tensor(self, spec, intensors=None):
        '''
        Parses a line from config file to make a tensor.
        '''
        spec = spec.strip().split()
        name = spec[0].strip(self.marker[0])
        double_comma = re.compile(',,')  # fix for baffling error
        function_spec = ''.join(spec[1:len(spec)])
        function_params = function_spec.split('(')
        func = function_params[0]
        if name in self._tensordict:
            return self._tensordict[name]
        elif len(spec) > 1:
            params = function_params[1].strip(')')
            if intensors is not None:
                params = double_comma.sub(',', 'intensors,' + params + ',name=name')
                params = params.strip(',')
            else:
                params = double_comma.sub(',', params + ',name=name')
                params = params.strip(',')
            if func == 'placeholder':
                return self._process_placeholder(func, params, name)
            elif func == 'lookup':
                return self._process_lookup(func, params, name)
            else:
                function_call = func + '(' + params + ')'
                try:
                    self._tensordict[name] = eval(function_call, NODE_GLOBALS, locals())
                    if self.develop:
                        heading = 'Node %s: %s' % (name, self.tensordict[name] )
                        print(termcolor.colored(heading, 'green'))
                        print('\tFunction Call: %s\n\tTensor Inputs:\n\t\t' %
                              (function_call), end="")
                        if type(intensors) is list:
                            print(*intensors, sep='\n\t\t')
                        else:
                            print(intensors)
                except NameError as e:
                    traceback.print_exc()
                    print(termcolor.colored("==========================Original Handled Exception Above | "
                                            "Input Tensors Below============================", 'red'))
                    print('Input Tensors:\n\t', end="")
                    if type(intensors) is list:
                        print(*intensors, sep='\n\t')
                    else:
                        print(intensors)
                    raise NameError('\nFunction Call: %s\n intensors: %r' % (function_call, intensors))
                except TypeError as e:
                    traceback.print_exc()
                    print(termcolor.colored("==========================Original Handled Exception Above |"
                                            "Input Tensors Below============================", 'red'))
                    print('Input Tensors:\n\t', end="")
                    if type(intensors) is list:
                        print(*intensors, sep='\n\t')
                    else:
                        print(intensors)
                    raise TypeError('\nFunction Call: %s\n intensors: %r' % (function_call, intensors))
                except ValueError as e:
                    traceback.print_exc()
                    print(termcolor.colored("==========================Original Handled Exception Above |"
                                            "Input Tensors Below============================", 'red'))
                    print('Input Tensors:\n\t', end="")
                    if type(intensors) is list:
                        print(*intensors, sep='\n\t')
                    else:
                        print(intensors)
                    raise ValueError('\nFunction Call: %s\n intensors: %r' % (function_call, intensors))
            return self._tensordict[name]
        else:
            raise MissingTensorError('Name %s: from config file is not in the tensor or placeholder '
                                     'dictionary so it must have a function call.' % name)

    def _get_edges(self, graph):
        '''
        Gets subgraphs and names of parent nodes.
        '''
        level = self._get_level(graph[0])
        node_names = [graph[0].strip().split()[0].strip(self.marker[0])]
        list = [0]
        for i in range(1, len(graph)):
            if self._get_level(graph[i]) == level:
                list.append(i)
                node_names.append(graph[i].strip().split()[0].strip(self.marker[0]))
        list.append(len(graph))
        intensors = []
        for i in range(0, len(list) - 1):
            intensors.append(graph[list[i]:list[i+1]])
        return intensors, node_names

    def _get_level(self, line):
        '''
        Find level of line from graph markers.
        '''
        spot = 0
        level = 0
        line = line.strip()
        while line[spot] == self.marker[0]:
            level += 1
            spot += 1
        if level % len(self.marker) != 0:
            raise GraphMarkerError('Need multiples of %s %s to delimit edges. Line: %s'
                                   % (len(self.marker), self.marker, line))
        return level

    def _process_placeholder(self, func=None, params=None, name=None):
        """
        Special treatment for placeholders which may be data dependent.
        """
        if name not in self._placeholderdict:
            if self.data is not None and name in self.data:
                params += ', data=self.data[name]'
            else:
                raise MissingDataError('There is no data called %s in the DataSet for this AntGraph.' % name)
            function_call = func + '(' + params + ')'
            try:
                self._placeholderdict[name] = eval(function_call, NODE_GLOBALS, locals())
                if self.develop:
                    heading = 'Node %s: %r' % (name, self.placeholderdict[name])
                    print(termcolor.colored(heading, 'green'))
                    print('\tFunction Call: %s\n\tInput Data: %r' %
                          (function_call, self.data[name]))
                return self._placeholderdict[name]
            except NameError as e:
                traceback.print_exc()
                print(termcolor.colored("==========================Original Handled Exception Above============================", 'red'))
                raise NameError('\nFunction Call: %s\nname=%r\ndata: %r hash=%s)' %
                                (function_call, name, self.data[name], name))
            except TypeError as e:
                traceback.print_exc()
                print(termcolor.colored("==========================Original Handled Exception Above============================", 'red'))
                raise TypeError('\nFunction Call: %s\nname=%r\ndata: %r hash=%s)' %
                                (function_call, name, self.data[name], name))
            except ValueError as e:
                traceback.print_exc()
                print(termcolor.colored("==========================Original Handled Exception Above============================", 'red'))
                raise ValueError('\nFunction Call: %s\nname=%r\ndata: %r hash=%s)' %
                                (function_call, name, self.data[name], name))
        return self._placeholderdict[name]

    def _process_lookup(self, func=None, params=None, name=None):
        """
        Special treatment for lookup function which may be data dependent.
        """
        paramlist = params.split(',')
        dataname = None
        for p in paramlist:
            if p.startswith('dataname='):
                dataname = p.split('=')[1]
        if dataname is not None:
            dataname = dataname.strip("'")
        if dataname in self._placeholderdict:
            params += ', makeplace=False, indices=self._placeholderdict[dataname], ' \
                      'data=self.data[dataname]'
        elif dataname in self.data:
            params += ', data=self.data[dataname]'
        else:
            function_call = func + '(' + params + ')'
            raise ProcessLookupError('"%s" is not a key in the data dictionary for this AntGraph. '
                                     'Need to provide a valid dataname argument for lookup without tensor input.'
                                     ' \nCall: %s' % (dataname, function_call))
        function_call = func + '(' + params + ')'
        try:
            vals = eval(function_call, NODE_GLOBALS, locals())
            self._tensordict[name] = vals[0]
            self._tensordict[name + '_weights'] = vals[1]
            self._placeholderdict[dataname] = vals[2]
            if self.develop:
                heading = 'Node %s: %s' % (name, self.tensordict[name])
                print(termcolor.colored(heading, 'green'))
                print('\tFunction Call: %s\n\tPlaceholder: %s\n\tWeights: %s\n\tInput Data: %r\n\t' %
                      (function_call, ph_rep(vals[1]), vals[2], self.data[dataname]))
            return self._tensordict[name]
        except NameError as e:
            traceback.print_exc()
            print(termcolor.colored("==========================Original Handled Exception Above============================", 'red'))
            raise NameError('\nFunction Call: %s\nname=%r\ndata: %r hash=%s)' %
                            (function_call, name, self.data[dataname], dataname))
        except TypeError as e:
            traceback.print_exc()
            print(termcolor.colored("==========================Original Handled Exception Above============================", 'red'))
            raise TypeError('\nFunction Call: %s\nname=%r\ndata: %r hash=%s)' %
                            (function_call, name, self.data[dataname], dataname))
        except ValueError as e:
            traceback.print_exc()
            print(termcolor.colored("==========================Original Handled Exception Above============================", 'red'))
            raise ValueError('\nFunction Call: %s\nname=%r\ndata: %r hash=%s)' %
                            (function_call, name, self.data[dataname], dataname))

    def _add_nodes(self, node_names):
        """
        Add nodes to graphviz dot string.
        """
        for node in node_names:
            self._dotstring += '\n\t' + node + ';'

    def _add_edges(self, start, dest):
        """
        Add edges to graphviz dot string.
        """
        self._dotstring += '\n\t' + start + ' -> {'
        for node in dest:
            self._dotstring += node + ','
        self._dotstring = self._dotstring.strip(',')
        self._dotstring += '} [dir=back];'

    def _substitute_variables(self, graph_spec, variable_bindings):
        """
        String substitution of graph text marked as variables.
        """
        has_marker = False
        test_graph = graph_spec.split('\n')
        for line in test_graph:
            if line.strip().startswith(self.marker):
                has_marker = True
        if not has_marker:
            raise GraphMarkerError("There are no instances of the chosen "
                                   "marker '%s' in the graph config file." % self.marker)
        if variable_bindings is None:
            indice = graph_spec.find('$')
            if indice >= 0:
                raise UndefinedVariableError('Need variable_bindings argument in call to AntGraph to bind '
                                             'variable beginning: %s' % graph_spec[indice:indice+10])
        else:
            for symbol in variable_bindings:
                replacee = '$' + symbol
                if graph_spec.find(replacee) >= 0:
                    if type(variable_bindings[symbol]) is str:
                        graph_spec = graph_spec.replace(replacee, "'" + str(variable_bindings[symbol] + "'"))
                    else:
                        graph_spec = graph_spec.replace(replacee, str(variable_bindings[symbol]))
                else:
                    raise UndefinedVariableError('%s is not mentioned in config file.' % replacee)
            indice = graph_spec.find('$')
            if indice >= 0:
                variable = graph_spec[indice+1:len(graph_spec)]
                if variable.find(',') >= 0:
                    variable = variable.split(',')[0]  #parameter in middle of function call
                elif variable.find(')') >= 0:
                    variable = variable.split(')')[0]  #parameter at end of function call
                else:
                    raise RandomNodeFunctionError('You forgot a parenthesis.')
                raise UndefinedVariableError('%s was not bound. Include %s in '
                                             'variable_bindings dictionary' % (variable, variable))
        return graph_spec

    def _import_node_files(self, files):
        '''
        Import node functions from modules in import parameter of constructor.
        '''
        for name in files:
            try:
                if files[name] is not None:
                    sys.path.append(files[name])
                m = __import__(name=name, globals=globals(), locals=locals(), fromlist="*")
                try:
                    attrlist = m.__all__
                except AttributeError:
                    attrlist = dir(m)
                for attr in [a for a in attrlist if '__' not in a]:
                    NODE_GLOBALS[attr] = getattr(m, attr)
            except ImportError, e:
                sys.stderr.write('Unable to read %s/%s.py\n' % (files[name], name))
                sys.exit(1)

# ====================================================================
# ===========Graph Format Testing ====================================
# ====================================================================
def testGraph(config, marker='-', graph_dest='antpics/', graph_name='test_graph'):
    """

    :param config: A graph specification in .config format.
    :param marker: A character or string of characters to delimit graph edges.
    :param graph_dest: Where to save the graphviz pdf and associated dot file.
    :param graph_name: A name for the graph (without extension)
    """
    with open(config, 'r') as config_file:
        graph_spec = config_file.read().strip().split('\n')  # remove whitespace at end of file
    outputs, node_names = _get_edges(graph_spec, marker)
    dotstring = 'digraph test_graph' + ' {'
    dotstring = _add_nodes(node_names, dotstring)
    for subgraph in outputs:
        dotstring = _traverse_graph(subgraph, marker, dotstring)
    dotstring += '\n}'
    if not graph_dest.endswith('/'):
        graph_dest += '/'
    path_to_graph_pic = graph_dest + graph_name + '.pdf'
    os.system('mkdir ' + graph_dest)
    with open(graph_dest + graph_name + '.dot', 'w') as dot_file:
        dot_file.write(dotstring)
    os.system('dot -Tpdf -o ' + path_to_graph_pic + ' ' + graph_dest + graph_name + '.dot')
    os.system('okular ' + path_to_graph_pic + ' &')

def _traverse_graph(graph, marker, dotstring):
    """
    This is a postorder 'tree' traversal with possibly repeated non-looping nodes.
    """
    if len(graph) == 1:
        return dotstring
    else:
        vertex_name = graph[0].strip().split()[0].strip(marker[0])
        edges, node_names = _get_edges(graph[1:len(graph)], marker)
        dotstring = _add_edges(vertex_name, node_names, dotstring)
        for end_node in edges:
            dotstring = _traverse_graph(end_node, marker, dotstring)
        return dotstring

def _get_edges(graph, marker):
    '''
    Gets subgraphs and names of parent nodes.
    '''
    level = _get_level(graph[0], marker)
    node_names = [graph[0].strip().split()[0].strip(marker[0])]
    list = [0]
    for i in range(1, len(graph)):
        if _get_level(graph[i], marker) == level:
            list.append(i)
            node_names.append(graph[i].strip().split()[0].strip(marker[0]))
    list.append(len(graph))
    intensors = []
    for i in range(0, len(list) - 1):
        intensors.append(graph[list[i]:list[i+1]])
    return intensors, node_names


def _get_level(line, marker):
        '''
        Find level of line from graph markers.
        '''
        spot = 0
        level = 0
        line = line.strip()
        while line[spot] == marker[0]:
            level += 1
            spot += 1
        if level % len(marker) != 0:
            raise GraphMarkerError('Need multiples of %s %s to delimit edges. Line: %s'
                                   % (len(marker), marker, line))
        return level

def _add_nodes(node_names, dotstring):
    """
    Add nodes to graphviz dot string.
    """
    for node in node_names:
        dotstring += '\n\t' + node + ';'
    return dotstring

def _add_edges(start, dest, dotstring):
    """
    Add edges to graphviz dot string.
    """
    dotstring += '\n\t' + start + ' -> {'
    for node in dest:
        dotstring += node + ','
    dotstring = dotstring.strip(',')
    dotstring += '} [dir=back];'
    return dotstring
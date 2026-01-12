"""
Copyright (c) 2021 Tilo van Ekeris / TMDT, University of Wuppertal
Distributed under the MIT license, see the accompanying
file LICENSE or https://opensource.org/licenses/MIT
"""
import win32com.client as win32
from .error_code import ErrorCode

class Plantsim:

    def __init__(self, version='', visible=True, trust_models=False, license_type='Professional'):
        print(f'\n+++ Launching Plant Simulation (version={version}, license={license_type}, visibility={visible}, trust_models={trust_models}) +++\n')

        dispatch_string = 'Tecnomatix.PlantSimulation.RemoteControl'
        if version != '':
            dispatch_string += f'.{version}'

        self.plantsim = win32.gencache.EnsureDispatch(dispatch_string)

        if visible:
            # Open the Plant Simulation window2
            self.plantsim.SetVisible(True)
        else:
            self.plantsim.SetVisible(False)


        if trust_models:
            # Allow Plant Simulation models to access the computer
            self.plantsim.SetTrustModels(True)

        self.license_type = license_type
        try:
            self.plantsim.SetLicenseType(self.license_type)
        except BaseException as e:
            if ErrorCode.extract(e.args) == -2147221503:
                raise Exception(f'The license type {self.license_type} does not seem to exist. Make sure it is a valid Plant Simluation license type.')

        self.license_type = ''
        self.path_context = ''
        self.event_controller = ''

    def load_model(self, filepath):

        print(f'Loading model "{filepath}"...\n')
        try:
            self.plantsim.LoadModel(filepath)
        except BaseException as e:
            if ErrorCode.extract(e.args) == -2147221503:
                raise Exception(f'The license server or the selected license type "{self.license_type}" is not available.\n'
                                + f'Make sure that the license server is up and running and you can connect to it (VPN etc.).\n'
                                + f'Make sure that a valid license of type "{self.license_type}" is available in the license server.')

    def set_event_controller(self):
        self.event_controller = f'{self.path_context}.Eventcontroller'

    def set_path_context(self, path_context):
        self.path_context = path_context
        self.plantsim.SetPathContext(self.path_context)

    def reset_simulation(self):
        if not self.event_controller:
            raise Exception('You need to set an event controller first!')

        self.plantsim.ResetSimulation(self.event_controller)

    def start_simulation(self):
        if not self.event_controller:
            raise Exception('You need to set an event controller first!')

        self.plantsim.StartSimulation(self.event_controller)

    def get_value(self, object_name):
        return self.plantsim.GetValue(object_name)

    def set_value(self, object_name, value):
        self.plantsim.SetValue(object_name, value)

    def execute_simtalk(self, command_string, parameter=None, from_path_context=True):
        if from_path_context:
            command_string = f'{self.path_context}.{command_string}'
        else:
            command_string = f'.{command_string}'

        if parameter:
            self.plantsim.ExecuteSimTalk(command_string, parameter)
        else:
            self.plantsim.ExecuteSimTalk(command_string)

    def quit(self):
        self.plantsim.Quit()
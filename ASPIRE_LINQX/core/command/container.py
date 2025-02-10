# Pydantic and Typing imports
from pydantic import (
    model_validator,
    model_serializer,
    PrivateAttr, 
)
from typing import (
    Dict, 
    Any,
    Optional,
    List,
)
# LINQX imports
from ASPIRE_LINQX.core.command.base import (
    BaseRunCommand,
    BaseInfoCommand, 
    BaseDriverCommand,
)
# Docker imports
import docker
from docker import DockerClient
from docker.models.containers import Container
# Singularity imports
from spython.main import Client
from spython.instance import Instance
from spython.main.base import Client as SpythonClient
# Other imports
import json
import time
import re

class ContainerLibraryCommand(BaseInfoCommand):
    '''
    Description
    -----------
    Base class for library commands corresponding to containerized microservices. Contains information 
    on which image to access, which commands to run, if any external devices are required, etc. These commands
    can be converted to run commands to be sent to another LCU and built into a corresponding driver command.

    Attributes
    ----------
    ```
    image_name : str
    ```
    The name of the image which will be used to create containerized microservices
    ```
    start_commands : List[str] = []
    ```
    A list of possible commands that can be used to start the containerized microservice
    ```
    default_start_command : str = "/bin/sleep infinity"
    ```
    The start commmand which will be defaulted to if no other one is chosen by the user
    ```
    run_commands : List[str] = []
    ```
    A list of possible commands that can be used to perform actions inside the containerized microservice
    ```
    default_run_command : Optional[str] = None
    ```
    The run command which will be defaulted to if no other one is chosen by the user
    ```
    devices : List[str] = []
    ```
    A list of external devices (i.e. GPU) which are needed for the microservice container to function
    correctly
    
    Methods
    -------
    ```
    def to_run_command(
            self, 
            uuid: str, 
            save_vars: Optional[Dict[str, str]] = None,
            start_command = None,
            run_command = None,
            devices = None,
        ) -> BaseRunCommand:
    ```
    Builds a `RunCommand` object based on user inputs and library command assigned values
    ```
    def to_message(
            self, 
            uuid: str, 
            save_vars: Optional[Dict[str, str]] = None,
            start_command = None,
            run_command = None,
            devices = None,
            ) -> str:
    ```
    Builds a JSON formatted message of to be sent to other LCU's based on user inputs 
    and library command assigned values
    '''
    image_name: str
    start_commands: List[str] = ["/bin/sleep infinity"]
    default_start_command: str = "/bin/sleep infinity"
    run_commands: List[str] = []
    default_run_command: str | None = None
    devices: List[str] = []

    @model_validator(mode='after')
    def validate_library_command(self) -> 'ContainerLibraryCommand':
        # Ensure that the default start command is valid
        if self.default_start_command not in self.start_commands:
            raise ValueError(f"Default start command '{self.default_start_command}' not in list of start commands: {self.start_commands}")
        
        # Ensure that the default run command is valid
        if self.default_run_command not in self.run_commands:
            raise ValueError(f"Default run command '{self.default_run_command}' not in list of run commands: {self.run_commands}")
        return self

    def to_run_command(
            self, 
            uuid: str, 
            save_vars: Optional[Dict[str, str]] = None,
            start_command: Optional[str] = None,
            run_command: Optional[str] = None,
        ) -> BaseRunCommand:
        '''
        Description
        -----------
        Builds a `RunCommand` object based on user inputs and library command assigned values
        
        Parameters
        ----------
        ```
        uuid : str
        ```
        The UUID of the endpoint where the run command will be sent to
        ```
        save_vars: Optional[Dict[str, str]] = None
        ```
        A list of variables to save off if the corresponding driver command is 
        run in a workflow
        ```
        start_command : str
        ``` 
        The command which will be used to start the containerized microservice (if needed)
        ```
        run_command : str
        ```
        The command which will be used to run some functionality of the containerized microservice
        
        Returns
        -------
        ```
        run_command : BaseRunCommand
        ```
        A `RunCommand` object created based on user inputs and library command assigned values
        '''

        # Assign defaults
        if save_vars is None: save_vars = {}
        if start_command is None: start_command = self.default_start_command
        if run_command is None: run_command = self.default_run_command

        # Ensure that the start command is valid
        if start_command is not None and start_command not in self.start_commands:
            raise ValueError(f"Start command '{start_command}' not in list of start commands {self.start_commands}")
        
        # Ensure that the run command is valid
        if run_command is not None and run_command not in self.run_commands:
            raise ValueError(f"Run command '{run_command}' not in list of run commands {self.run_commands}")

        # Make sure all variables to save are in the return signature
        for key in save_vars.keys(): 
            if key not in self.return_signature.keys(): raise KeyError(f"Key: '{key}' is not a valid return key, expected one of: {self.return_keys}")

        # Make sure all variables in the run command are in parameters
        for elem in re.findall('\{.*?\}', self.run_command):
            if str(elem) not in self.parameters.keys():
                raise KeyError(f"Run command variable '{elem}' not found in parameters: {self.parameters.keys()}")
            
        return BaseRunCommand(
            **{
                "uuid": uuid,
                "name": self.name,
                "parameters": self.parameters.copy(),
                "save_vars": save_vars.copy(),
                "image_name": self.image_name,
                "start_command": start_command,
                "run_command": run_command,
                "devices": self.devices,
            }
        )
    
    def to_message(
            self, 
            uuid: str, 
            save_vars: Optional[Dict[str, str]] = None,
            start_command: str = None,
            run_command: str = None,
        ) -> str:
        '''
        Description
        -----------
        Builds a JSON formatted message of to be sent to other LCU's based on user inputs 
        and library command assigned values
        
        Parameters
        ----------
        ```
        uuid : str
        ```
        The UUID of the endpoint where the run command will be sent to
        ```
        save_vars: Optional[Dict[str, str]] = None
        ```
        A list of variables to save off if the corresponding driver command is 
        run in a workflow
        ```
        start_command : str
        ``` 
        The command which will be used to start the containerized microservice (if needed)
        ```
        run_command : str
        ```
        The command which will be used to run some functionality of the containerized microservice

        Returns
        -------
        ```
        message : str
        ```
        A JSON formatted message of to be sent to other LCU's based on user inputs 
        and library command assigned values
        '''
        return self.to_run_command(
            uuid=uuid,
            save_vars=save_vars,
            start_command=start_command,
            run_command=run_command,
        ).m_json()
        
class ContainerDriverCommand(BaseDriverCommand):
    '''
    Description
    -----------
    Base class for execution of containerized microservices. This class is not desiged to implemented
    execution of containerized microservices, but instead is designed to be extended by classes
    which can execute containerized microservices using a specific platform. Thus many of the methods 
    serve as common placeholders and remain unimplemented.

    Attributes
    ----------
    ```
    uuid : str
    ```
    UUID of system associated with the driver command
    ```
    image_name : str
    ```
    The name (including tag) of the image that will be used to create containers associated
    with this command
    ```
    devices : Optional[List[Any]]
    ```
    External devices (such as GPU) which will be provided to containers associated with this
    command
    ```
    run_infinity: bool = True
    ```
    A flag to run the container infinitly upon command `__call__`. 
    
    If this flag is set to True, when the command is called, if a container is not currently 
    assigned to the command, `self.start()` will be called and the run command will be executed
    in the container. `self.stop()` and `self.remove()` will not be called after command execution
    and thus the container will remain running.

    If this flag is set to False, after the command is called `self.stop()` and `self.remove()`
    will be called to stop and remove the container after execution.
    ```
    run_command : Optional[str]
    ```
    The command which will be run in the container upon command `__call__` override
    ```
    start_delay : float
    ``` 
    How long to delay after starting a containerized microservice to allow for setup.
    ```
    has_return : Optional[bool] = False
    ```
    True if function has a return value and it should be accessed, False otherwise

    Methods
    -------
    ```
    def start(self) -> None:
    ```
    Starts a container of the provided image name (`self.image_name`) and assigns it to the 
    commands container (`self._container`) for execution of the microservice.
    ```
    def stop(self) -> None:
    ```
    Stops the container assigned to the command (`self._container`)
    ```
    def remove(self) -> None:
    ```
    Stops and removes the container assigned to the command (`self._container`)
    '''
    
    # Public attributes
    image_name: str
    devices: List[Any] | None = None
    run_infinity: bool = True
    start_delay: float = 2.0

    # These are not needed in a containerized driver command
    fn: None = None
    module: None = None
    package: None = None

    @model_validator(mode='after')
    def validate(self) -> 'ContainerDriverCommand':
        return self
    
    def _assign_default_container_kwargs(self) -> Dict[str, Any]:
        '''
        Description
        -----------
        Designed to return a dictionary of default arguments which are used to
        start containerized microservices.

        Not implemented in the base class
        '''
        raise NotImplementedError

    def start(self) -> None:
        '''
        Description
        -----------
        Designed to start a containerized microservice belonging to the image
        defined in the command.
         
        Not implemented in the base class
        '''
        raise NotImplementedError
    
    def stop(self) -> None:
        '''
        Description
        -----------
        Designed to stop the containerized microservice assigned to this command.

        Not implemented in the base class
        '''
        raise NotImplementedError
    
    def remove(self) -> None:
        '''
        Description
        -----------
        Designed to remove the containerized microservice assigned to this command.

        Not implemented in the base class
        '''
        self.stop()
        raise NotImplementedError

    def __call__(self, wf_globals: Dict[str, Any]=None, save_vars: Dict[str,str]=None, container_kwargs: Dict[str, Any]=None, **kwargs) -> Dict:
        '''
        Description
        -----------
        Run the specified microservice image as a Docker containerized microservice. 
        Executes the `run_container` function which behaves in the following manner:
        ```
        # If run_infinity is set to False
        def run_container():
            # Run container of specified image with docker_kwargs
            # Upon container exit, save the logs and remove the container
            # Return the logs as a dictionary if has_return is True
        # If run_infinity is set to True
        def run_container():
            # Search running containers for one with image_name match
            if image_name in client.containers.list():
                # Execute the command and return the output if has_return is True
            else:
                # Run a new container with /bin/sleep infinity
                # Execute the command and return the output if has_return is True
        ```

        Parameters
        ----------
        ```
        wf_globals : Dict[str, Any] = None
        ```
        Dynamic arguments to call the driver command function with, must match with `self.parameter`.
        Specified results are saved to this dictionary after command execution.
        ```
        save_vars : Dict[str,str] = None
        ```
        Dictionary of varaibles to save off from command output to wf_globals
        ```
        docker_kwargs : Dict[str, Any]
        ```
        Arguments when running the docker container -> `client.containers.run(**docker_kwargs)`. 
        
        By default, docker kwargs is set to the following (you may override if needed):
        ```
        docker_kwargs = {
            "image": self.image_name,                                   # Provided image and tag
            "command": self._set_command(self._parameters_to_envs()),   # Command with parameters (as env vars) replaced
            "detach": True,                                             # Run in detached mode
            "environment": self._parameters_to_envs(),                  # Provide parameters as env vars
            "device_requests": self.devices                             # Provided requested devices (i.e. GPU)
        }
        ```
        ```
        kwargs: Dict[str, Any]
        ```
        Static arguments to call the driver command function with. Must match with `self.parameters`

        Returns
        -------
        ```
        result : Dict[str, Any]
        ```
        A dictionary result of the microservice container execution
        '''
        
        # Set defaults for the dictionaries
        if save_vars is None: save_vars = {}
        if wf_globals is None: wf_globals = {}

        # Validate parameters
        self._validate_kwargs(**kwargs)

        # Update parameters
        self._update_parameters(wf_globals, **kwargs)
            
        # Run the funciton -> call docker.run()
        result = self._function(self.has_return, container_kwargs)
        if self.has_return:
            # Save the result to globals
            self._save_results_to_globals(result, wf_globals, save_vars)
            return result
    
class DockerDriverCommand(ContainerDriverCommand):
    '''
    Description
    -----------
    Driver command for execution of a containerized microservice as a Docker container.

    Attributes
    ----------
    ```
    uuid : str
    ```
    UUID of system associated with the driver command
    ```
    image_name : str
    ```
    The name of the docker image which will be used.
    Format as `'<image_name>:<tag_name>'`
    ```
    devices : List[Any] | None = None
    ```
    External devices which can be provided to the microservice container at execution.
    See Docker Python SDK for more details.
    ```
    run_infinity: bool = True
    ```
    A flag to run the container infinitly upon command `__call__`. 
    
    If this flag is set to True, when the command is called, if a container is not currently 
    assigned to the command, `self.start()` will be called and the run command will be executed
    in the container. `self.stop()` and `self.remove()` will not be called after command execution
    and thus the container will remain running.

    If this flag is set to False, after the command is called `self.stop()` and `self.remove()`
    will be called to stop and remove the container after execution.
    ```
    run_command : Optional[str]
    ```
    The command which will be run in the container upon command `__call__` override
    ```
    start_command : str = '/bin/sleep infinity'
    ```
    The command which will be run upon starting a container. Generally, this command should 
    be set to something that keeps the container running infinitely.
    ```
    start_delay : float
    ``` 
    How long to delay after starting a containerized microservice to allow for setup.
    ```
    has_return : Optional[bool] = False
    ```
    True if function has a return value and it should be accessed, False otherwise

    Methods
    -------
    ```
    def start(self) -> None:
    ```
    Starts a container of the provided image name (`self.image_name`) and assigns it to the 
    commands container (`self._container`) for execution of the microservice.
    ```
    def stop(self) -> None:
    ```
    Stops the container assigned to the command (`self._container`)
    ```
    def remove(self) -> None:
    ```
    Stops and removes the container assigned to the command (`self._container`)
    ```
    def __call__(
        wf_globals: Dict[str, Any]=None, 
        save_vars: Dict[str,str]=None, 
        docker_kwargs: Dict[str, Any]=None, **kwargs
    ) -> Dict:
    ```
    Executes the `run_container` function which behaves in the following manner:
    ```
    # If run_infinity is set to False
    def run_container():
        # Run container of specified image with docker_kwargs
        # Upon container exit, save the logs and remove the container
        # Return the logs as a dictionary if has_return is True
    # If run_infinity is set to True
    def run_container():
        # Search running containers for one with image_name match
        if image_name in client.containers.list():
            # Execute the command and return the output if has_return is True
        else:
            # Run a new container with /bin/sleep infinity
            # Execute the command and return the output if has_return is True
    ```
    By default, docker kwargs is set to the following (you may override if needed):
    ```
    docker_kwargs = {
        "image": self.image_name,                                   # Provided image and tag
        "command": self._set_command(self._parameters_to_envs()),   # Command with parameters (as env vars) replaced
        "detach": True,                                             # Run in detached mode
        "environment": self._parameters_to_envs(),                  # Provide parameters as env vars
        "device_requests": self.devices                             # Provided requested devices (i.e. GPU)
    }
    ```
    '''
    
    # Public attributes
    start_command: str = "/bin/sleep infinity"
    run_command: str | None = None

    # Private attributes
    _client: DockerClient = PrivateAttr(default=None)
    _container: Container = PrivateAttr(default=None)

    @model_validator(mode='after')
    def validate_docker_command(self) -> 'DockerDriverCommand':
        # TODO: validate 
        from warnings import warn
        warn('Docker commands are currently not supported', DeprecationWarning)
        return self
    
    def _init_private_attributes(self):
        '''
        Assign docker client and container run function at object creation
        '''
        # Assign _client from env
        self._client = docker.from_env()

        def run_container(has_return: bool, container_kwargs: Dict[str, Any] = None) -> Optional[Dict]:
            # TODO: This will need to be edited to be compatible with docker volume
            # If there is no current container bound to the command
            if self._container is None: self.start(docker_kwargs=container_kwargs)

            # Run the command using exec_run and decode the output
            exit_code, output = self._container.exec_run(
                self._set_command(self._parameters_to_envs()),
                environment=self._parameters_to_envs()
                )
            output = output.decode()

            # If we are not running the container infinitely, stop it
            if not self.run_infinity: self.stop()

            # Return the output
            if has_return: json_output = json.loads(output); return json_output

        self._function = run_container

    def _assign_default_container_kwargs(self) -> Dict[str, Any]:
        '''
        Descirption
        -----------
        Returns default conatiner arugments for microservices run using Docker

        Returns
        -------
        container_kwargs : dict[str, Any]
        Default container arguments for Docker microservices
        ```
        return {
            "image": self.image_name,
            "command": self.start_command,
            "detach": True,
            "device_requests": self.devices,
        }
        ```
        '''
        return {
            "image": self.image_name,
            "command": self.start_command,
            "detach": True,
            "device_requests": self.devices,
        }

    def _parameters_to_envs(self) -> Dict[str, str]:
        '''
        Converts command parameters to a dictionary of enviroment varailbes for use in `docker.run()`

        ```
        return env_dict : Dict[str, str]
        ```
            A dictionary represention of the parameters with capatilized keys and string values
        '''
        env_dict = {}
        for key, param in self.parameters.items():
            env_dict[key.upper()] = str(param.value)
        return env_dict

    def _set_command(self, env_vars: Dict[str, Any]) -> str:
        '''
        Description
        -----------
        Replaces the command to be executed in the microservice container 
        with driver command parameters (based on env var assignment). 

        Example
        ------- 
        ```
        env_vars = {"ENV1": 1, "ENV2": "Two"}
        self.run_command = "python example.py --arg1 {ENV1} --arg2 {ENV2}"
        command = self._set_command(env_vars) # Note that self.run_command is currently not updated
        ```
        Will run the following in the container
        ```bash
        >>> python example.py --arg1 "1" --arg2 "Two"
        ```

        Parameters
        ----------
        ```
        env_vars : Dict[str, Any]
        ```
        A dictionary of enviroment varaibles corresponding to runtime parameters of the
        containerized microserivce

        Return
        ------
        ```
        command : str
        ```
        The command with varaibles replaced from the enviroment varaible dictionary. 
        Note that `self.run_command` is not updated
        '''
        command = self.run_command
        for key, value in env_vars.items():
            placeholder = "{" + key + "}"
            command = command.replace(placeholder, f'\"{str(value)}\"')
        return command

    def start(self, container_kwargs: Dict[str, Any] = None) -> Container:
        '''
        Description
        -----------
        Starts a containerized microservice belonging to the image
        defined in the command using Docker. If there is no current 
        container assigned to the command, starts a new container 
        and sets `self._container` to the container which was started.
        Otherwise starts the container assigned to this command.

        Parameters
        ----------
        ```
        docker_kwargs : Dict[str, Any] = None
        ```
        Kwargs to be called when starting the Docker container.
        Defaults are as follows:
        ```
        docker_kwargs = {
            "image": self.image_name,
            "command": self._set_command(self._parameters_to_envs()),
            "detach": True,
            "environment": self._parameters_to_envs(),
            "device_requests": self.devices,
            "command": self.start_command,
        }

        Return
        ------
        ```
        container : Container
        ```
        The container that was started by this method. Also assigned to 
        `self._container`
        ```
        '''
        if container_kwargs is None: container_kwargs = self._assign_default_container_kwargs()

        if self._container is None:
            self._container = self._client.containers.run(**container_kwargs)
            time.sleep(self.start_delay)
        else: 
            self._container.start()

        return self._container

    def stop(self) -> None:
        '''
        Description
        -----------
        If the Docker container corresponding to this command (`self._container`) is running,
        stop the container.
        '''
        if self._container.status == "running": self._container.stop()
        
    def remove(self) -> None:
        '''
        Description
        -----------
        Removes the Docker container corresponding to this command.
        '''
        self._container.remove(force=True)
        self._container = None
        
class ApptainerDriverCommand(ContainerDriverCommand):
    '''
    Description
    -----------
    Driver command for execution of a containerized microservice as a Singularity container.

    Attributes
    ----------

    Methods
    -------
    ```
    def start(self) -> None:
    ```
    Starts a container of the provided image name (`self.image_name`) and assigns it to the 
    commands container (`self._container`) for execution of the microservice.
    ```
    def stop(self) -> None:
    ```
    Stops the container assigned to the command (`self._container`)
    ```
    def remove(self) -> None:
    ```
    Stops and removes the container assigned to the command (`self._container`)
    ```
    '''
    
    # Public attributes
    run_script: bool = False
    run_app: bool = False
    app: str | None = None
    verbose: bool = False
    gpu: bool = False

    # Private attributes
    _client: SpythonClient = PrivateAttr(default=None)
    _instance: Instance = PrivateAttr(default=None)

    @model_serializer(when_used='json')
    def serialize_apptainer_driver_command(self) -> Dict:
        return self.model_dump(exclude_none=True)

    def _validate_parameters(self) -> None:
        pass

    @model_validator(mode='after')
    def validate(self) -> 'ApptainerDriverCommand':
        # TODO add validation
        self._init_private_attributes()
        return self
    
    def _assign_default_container_kwargs(self) -> Dict[str, Any]:
        '''
        Description
        -----------
        Provides a dictionary of default apptainer instance run kwargs

        Return
        ------
        ```
        container_kwargs : Dict[str, Any]
        ```
        Runtime arguments apptainer instance run or command execution
        '''
        container_kwargs = {
            "image": self._instance,
            "options": [
                "--cleanenv",
            ],
            "nv": self.gpu,
            "return_result": True,
            "quiet": False,
        }
        if self.run_app: container_kwargs["app"] = self.app
        if not self.run_script and not self.run_app: container_kwargs["command"] = self.run_command
        if self._parameters is not None and len(self._parameters.keys()) > 0: 
            container_kwargs["options"].append(f"--env={self._parameters_to_envs()}")
        return container_kwargs

    def _parameters_to_envs(self) -> str:
        '''
        Description
        -----------
        Converts driver command parameters to a string of environment varaibles
        compatible with apptainer execute command
        
        Return
        ------
        A string of environment varaibles compatible with apptainer execute command
        ```
        return "VAR1=A,VAR2=B,VAR3=C"
        ```
        '''
        return ",".join(
            [
                f"{name.upper()}={str(param.value).replace(' ', '')}" for name,param in self._parameters.items()
            ]
        )

    def _init_private_attributes(self):
        '''
        Description
        -----------
        Initialize Singularity run command at object creation (`self._function`)
        '''
        super()._init_private_attributes()
        self._client = Client
        # This will prevent the client from printing output if not verbose mode
        if not self.verbose: self._client._init_level(True)
        if self.verbose: self._client._init_level(False)
        self._instance = None
        
        def run_instance(has_return: bool, container_kwargs: Dict[str, Any] = None):
            # If there is no container currently assigned to the command, start one
            # NOTE: spython client appears to be broken for running
            if self._instance is None: self.start()
            
            # Run the command and get the output based on run config
            if container_kwargs is None: container_kwargs = self._assign_default_container_kwargs()

            if self.run_script or self.run_app: output = self._client.run(**container_kwargs)
            else: output = self._client.execute(**container_kwargs)

            # Make sure that our output is formatted correctly
            if not isinstance(output, dict):
                raise TypeError(f"Invalid return of spython client, expected {dict} but received {type(output)}")
            if output["return_code"] != 0:
                return output

            # Get the output message
            output = output["message"]

            # Stop the container if it is not supposed to run infinitely
            if self.run_infinity is False: self.stop()

            # Return command output if specified
            if has_return: json_logs = json.loads(output); return json_logs
        
        self._function = run_instance

    def start(self) -> Instance:
        '''
        Description
        -----------
        Starts a containerized microservice belonging to the image
        defined in the command using Singularity. If there is no current 
        container assigned to the command, starts a new container 
        and sets `self._container` to the container which was started.
        Otherwise starts the container assigned to this command. A start 
        delay is included to allow for setup.

        Parameters
        ----------

        Return
        ------
        ```
        container : Instance
        ```
        The container that was started by this method. Also assigned to 
        `self._container`
        ```
        '''
        if isinstance(self._instance, Instance):
            self.stop()
        options = []
        if self.gpu: options.append("--nv")
        self._instance = self._client.instance(self.image_name, options=options)
        time.sleep(self.start_delay)
        return self._instance

    def stop(self) -> None:
        '''
        Description
        -----------
        If the Singularity container corresponding to this command (`self._container`) is running,
        stop the container.
        '''
        if self._instance is not None:
            self._instance.stop()
            self._instance = None
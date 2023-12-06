import sys
import pathlib
import pkgutil
import importlib
import inspect
import logging

from . import DataProvider
from . import ConfigManager





class DataProviderFactory:

    def __init__(self, providers_folder: str=None, providers_list: list=None, providers_extra_objects=None):
        # 1. Check if providers_folder is valid
        # 2. Load all DPs whose IDs are in providers_list. Load all if None
        # 3. If no providers_folder passed, search for config_file
        # 4. Look for providers_folder in config file
        # 5. Look for providers_list in config file
        # 6. Check for providers in xingu.dataproviders


        # Setup logging
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)


        # Initialize parameters to something useless and fill them up along the way
        self.providers_folder   = None
        self.providers_list     = None
        self.config             = None
        self.dps                = {}


        # Now overwrite whatever came from config file with what was passed to __init__
        if providers_folder is not None:
            self.providers_folder = providers_folder
        else:
            self.providers_folder = ConfigManager().get('DATAPROVIDER_FOLDER', default=None)

        if providers_list is not None:
            self.providers_list = providers_list
        else:
            self.providers_list = ConfigManager().get('DATAPROVIDER_LIST', default=None)

        if isinstance(self.providers_list, list):
            # Remove useless elements
            self.providers_list = list(filter(None, self.providers_list))
            if (len(self.providers_list) < 1):
                self.providers_list = None
        elif self.providers_list is not None:
            self.providers_list = [i.strip() for i in self.providers_list.split(',')]


        # If we still don't have a path to work with, use current folder
        if self.providers_folder is None:
            self.providers_folder = str(pathlib.Path('.'))

        self.logger.info("Folder to search DataProviders: {}".format(self.providers_folder))

        # We are now configured as much as we could.

        # Insert self.providers_folder in modules path in order to
        # make importlib.import_module() work properly
        sys.path.insert(1, self.providers_folder)


        # Load and store classes which IDs on self.providers_list
        for modinfo in pkgutil.iter_modules([self.providers_folder]):
            # Import the python file as a module
            mod=importlib.import_module(modinfo.name)
            self.logger.debug("Imported module {}".format(modinfo.name))

            # Walk through each thing inside that module and check if it is a desired DP
            for thing_name in dir(mod):
                thing = getattr(mod, thing_name)

                if (
                        # Check if class
                        inspect.isclass(thing)

                        # Check if subclass of DataProvider
                        and issubclass(thing, DataProvider)

                        # Check if we have and ID
                        and hasattr(thing, 'id')

                        # Check if ID is meaningful
                        and thing.id is not None

                        # Check if not DataProvider itself
                        and thing != DataProvider

#                         and (
#                             # Check if we have a list of desired classes, use it anyway otherwise
#                             self.providers_list is None or
#
#                             # Check if class is desired through its ID
#                             thing.id in self.providers_list
#                         )
                ):
                    self.dps.update({thing.id: thing})

        # Process DataProvider objects passed on providers_extra_objects
        if providers_extra_objects is not None:
            if issubclass(type(providers_extra_objects),DataProvider):
                providers_extra_objects=[providers_extra_objects]
            if isinstance(providers_extra_objects,list):
                self.dps.update({
                    dp.id: dp
                    for dp in providers_extra_objects
                })

        if self.providers_list is None:
            self.providers_list=list(self.dps.keys())



    def produce(self):
        for dp in self.providers_list:
            # Return a concrete class instance of a true real DataProvider implementation
            try:
                if type(self.dps[dp])==type:
                    yield self.dps[dp]()
                else:
                    yield self.dps[dp]
            except KeyError as e:
                self.logger.exception(f'Can’t find DataProvider with ID «{dp}».')



    def get_pre_req(self, dp):
        """
        Return a set of IDs of all pre-req models for dp.
        So you'll get {cartorios,anuncios} for dp=vendas_loft.
        """
        pre_req=set()
        sub_pre_req=set()

        if isinstance(dp, str):
            # Convert a DataProvider ID into a real DataProvider
            dp=self.dps[dp]

        if hasattr(dp, 'pre_req'):
            pre_req=pre_req | dp.pre_req

            if len(pre_req):
                for p in pre_req:
                    sub_pre_req=sub_pre_req | self.get_pre_req(p)

        return pre_req | sub_pre_req



    def __repr__(self):
        return 'DataProviderFactory(' + self.dps.__repr__() + ')'
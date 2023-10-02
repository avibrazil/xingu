import os
import re
import decouple


# Singleton implementation from https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
# method 3
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigManager(object, metaclass=Singleton):
    """
    A singleton class for all Xingu configuration affairs.

    It is pseudo-singleton because this is just emulated as all-static methods class. So
    you are not supposed to instantiate object of this class, just use its methods
    prefixing the class name, as:

    ConfigManager.get('VARIABLE', default='something')

    Methods are capable of recursively resolving AWS Parameter Store and AWS Secrets as:

    • {% AWS_PARAM:some-parameter %}
    • {% AWS_SECRET:some-secret %}

    So if an environment variable named TRAINED_MODELS_PATH has value
    "s3://{%AWS_PARAM:robson-avm-staging-bucket%}/trained-models", the get() method will
    search for all {% AWS_PARAM:some-parameter %} and {% AWS_SECRET:some-secret %}
    inside the value and recursively access these services to resolve them to their actual
    values.

    Pretty practical because it lets you write cleaner and concise configurations for
    Xingu as seen in .github/workflows/build_and_train_staging.yml section ‘Execute
    application’.
    """


    undefined       = decouple.undefined

    cache           = {}


    def __init__(self, envfile='./.env'):
        try:
            self.config=decouple.Config(decouple.RepositoryEnv(envfile))
        except FileNotFoundError as e:
            if envfile == './.env':
                self.config = decouple.config
            else:
                raise e



    def get(self, config_item: str, default=decouple.undefined, cast=decouple.undefined, resolve=True, value_only=True):
        """
        Convert an environment variable (or any repository suported by decouple module)
        or AWS Parameter or AWS Secret into its value.

        Use ‘AWS_SECRET:’ or ‘AWS_PARAM:’ as prefix to retrieve values from these systems.

        Use ‘{% VARIABLE %}’ to replace value for env var into a text snipped.

        Here is a full example:

        'postgresql+{% ROBSON_DB_PGSQL_DRIVER %}://{% AWS_PARAM:robson-avm-staging-user %}:{% AWS_SECRET:robson-avm-staging-rds-secret %}@{% AWS_PARAM:robson-avm-staging-url %}/{% AWS_PARAM:robson-avm-staging-database-name %}'

        Use spaces inside ‘{% %}’ to increase readability.

        If config_item is undefined and not found in environment, the value passed to
        default will be returned. If default is also undefined (decouple.undefined), an
        exception will be raised.

        If resolve==False, {% AWS_PARAM:... %} and {% AWS_SECRET:... %} will not be
        recursively resolved and left as is. Useful only in debug and development
        scenarios.

        If value_only==False, returns the entire dict provided by these AWS services. This
        is only useful in debug and development scenarios.
        """

        if config_item in self.cache:
            if cast==decouple.undefined:
                return self.cache[config_item]
            else:
                return cast(self.cache[config_item])

        elif config_item.startswith(('AWS_PARAM:','AWS_SECRET:')):
            import boto3

            if not hasattr(self,'parameter_store'):
                self.parameter_store = boto3.client('ssm')

            if not hasattr(self,'secrets_manager'):
                self.secrets_manager = boto3.client('secretsmanager')

            aws_item=config_item.split(':')[1]

            try:
                if config_item.startswith('AWS_PARAM:'):
                    value=self.parameter_store.get_parameter(Name=aws_item)
                    if value_only: return ConfigManager.set_cache(config_item,value['Parameter']['Value'])
                elif config_item.startswith('AWS_SECRET:'):
                    value=self.secrets_manager.get_secret_value(SecretId=aws_item)
                    if value_only: return ConfigManager.set_cache(config_item,value['SecretString'])
                return self.set_cache(config_item,value)
            except (
                self.parameter_store.exceptions.ParameterNotFound,
                self.secrets_manager.exceptions.ResourceNotFoundException
            ) as e:
                if default!=ConfigManager.undefined:
                    return default
                else:
                    raise decouple.UndefinedValueError(
                        '{} not found in AWS Parameters or Secrets services.'.format(aws_item)
                    )
        else:
            # Get the value of this config item from environment
            value=self.config(config_item, default=default, cast=cast)

            if value is not None and isinstance(value, str) and resolve:
                if '{%' in value:
                    # Recursively handle embedded variables

                    ## Detect all variables
                    placeholders={e for e in re.findall(r'\{\%.*?\%\}',value)}

                    for p in placeholders:
                        ## Iterate over each variable/placeholder and replace by its value
                        value=value.replace(
                            p,
                            self.get(
                                re.sub(r'\{\%|\%\}','',p).strip(),
                                cast=cast
                            ),
                        )
                elif value.startswith(('AWS_PARAM:','AWS_SECRET:')):
                    # Recursively handle AWS parameters ans secrets
                    value=self.get(value, cast=cast)

            if value == default:
                return value
            else:
                return self.set_cache(config_item,value)



    def set_cache(self, config_item, value):
        if value == ConfigManager.undefined and config_item in self.cache:
            del self.cache[config_item]
        else:
            self.cache[config_item] = value
        return value



    def set(self, config_item, value=undefined):
        """
        Set environment variables.

        If a simple name is passed, set it with value.

        If a list of names is passed, set all of them with value.

        If a dict is passed, set it as a whole in the environment.

        If value is ConfigManager.undefined, unset the variable(s).
        """
        if isinstance(config_item, dict):
            for c in config_item.keys():
                self.set(c, config_item[c])
        elif isinstance(config_item, list) or isinstance(config_item, tuple):
            for i in config_item:
                self.set(i, value)
        elif isinstance(config_item, str):
            self.set_cache(config_item, value)
            if value == ConfigManager.undefined:
                if config_item in os.environ:
                    del os.environ[config_item]
            else:
                if isinstance(value, str):
                    os.environ[config_item] = value
                elif value is None:
                    os.environ[config_item] = ''
                else:
                    os.environ[config_item] = str(value)

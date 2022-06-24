from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser()
config.read('config/config.ini')

class MyConfig():
    def __init__(self, config_file) -> None:
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read(config_file, encoding='utf-8')
        
        for section in self.config.sections():
            for k, v in self.config.items(section):
                k, v = self.get_type(k, v)
                self.__setattr__(k, v)
    
    def get_type(self, k, v):
        if v.lower() == 'true':
            v = True
        elif v.lower() == 'false':
            v = False
        elif v.lower() == 'none':
            v = None
        elif '[' == v[0] and ']' == v[-1]:
            v = v.replace('[', '')
            v = v.replace(']', '')
            v = v.split(',')
        else:
            try:
                v = eval(v)
            except:
                v = v
        return k, v
    
        
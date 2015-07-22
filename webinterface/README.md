# Web Interface Relefancer

This package, created with `NodeJS`, connects to the database of the Relevancer.

This README contains:

1. [Installation guide]
2. [Usage guide]

[Installation guide]: #installation
[Usage guide]: #usage

## Installation

This guide will help you to get this module up and running.


### Prerequisites

- [NodeJS]
- [Bower]
- [GruntJS]
- [MongoDB]
- [Git]
- [TweetsFetcher]

[NodeJS]: http://nodejs.org/
[Bower]: http://bower.io/#install-bower
[GruntJS]: http://gruntjs.com/getting-started
[MongoDB]: http://docs.mongodb.org/manual/installation/
[Git]: http://git-scm.com/book/en/Getting-Started-Installing-Git

### Now what?

After all prerequisites have been installed, it is time to get the source code. This can be done by:

1. [downloading] the source code.
2. Using the command: `git clone git@github.com:cengelif/Relevancer.git`.

[downloading]: archive/master.zip

From here on out, the installation guide will proceed inside the command line interface, so open a *Command Line Terminal* of your preference (or *CMD* on WINDOWS, make sure **node** and **npm** are part of your path) and navigate inside the source code root folder.
All commands need to be executed inside the source code root folder, unless stated otherwise, and will have the following format:
```
$ <<command here>>
```

#### Production

If you're just planning to use this module inside your own project, you only need to do a production install:
```
$ npm install --production
```

After all the dependencies have been resolved, go ahead and install all the bower componants:
```
$ bower install --production
```

Finally, the code needs to be compiled. You can do this by simply running the grunt command in the root folder, like so:
```
$ grunt
```
And that's it! Now your ready to use this module.

#### Development

If you want to help developing for this module, or are planning to modify this package in any way, it is recommended to follow a different installation process.
To set up the devolpment environment, you can simply execute the following commands:
```
$ npm install
$ bower install
```

Now you're ready to develop!
At any time, when you want to compile the code and let your latest version run, just execute:
```
$ grunt
```

If you want to run all the tests, you can just type `npm test`, and all tests will run automatically.
Now your are ready to roll, easy as pie!

## Usage
Once everything is ready to roll, you can start deploying:
```
$ node [path/to/project]
```

For more information about running the script in the background, please take a look at the [Forever] package

[Forever]: https://github.com/nodejitsu/forever

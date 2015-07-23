module.exports = function(grunt) {
   grunt.initConfig({
      pkg: grunt.file.readJSON('package.json'),

      watch: {
         options: {
            livereload: true
         },
         scsslint: {
            files: ['<%= scsslint.files %>'],
            tasks: ['scsslint', 'sass']
         },
         jshint: {
            files: ['<%= jshint.files %>', '.jshintrc'],
            tasks: ['jshint', 'uglify']
         },
         jade: {
            files: ['<%= sources.jade %>'],
            tasks: ['copy:jade']
         },
      },
      
      scsslint: {
         files: ['<%= sources.scss %>'],
         options: {
            config: '.scss-lint.yml'
         }
      },

      jshint: {
         files: ['<%= sources.gruntfile %>', '<%= sources.js %>'],
         options: {
            jshintrc: '.jshintrc'
         }
      },

      sass: {
         options: {
            // loadPath: ['bower_components/foundation/scss'],
            loadPath: ['bower_components/bootstrap-sass-official/assets/stylesheets', 'bower_components/bourbon/app/assets/stylesheets'],
            style: 'compressed',
            quiet: true,
         },
         dist: {
            files: [{
               expand: true,
               cwd: 'src/scss',
               src: ['*.scss'],
               dest: 'build/css',
               ext: '.css'
            }]
         }
      },

      uglify: {
         options: {
            mangle: {
               toplevel: true
            },
            report: 'gzip'
         },
         files: {
            src: ['<%= sources.js %>', 'src/js/ga.js'],
            dest: 'build/js',
            expand: true,
            flatten: true,
            ext: '.min.js'
         }
      },
      
      copy: {
         jade: {
            expand: true,
            cwd: 'src/jade',
            src: '**',
            dest: 'build/jade'
         },
         data: {
            expand: true,
            cwd: 'assets/',
            src: 'data/**',
            dest: 'build/',
         },
         fonts: {
            expand: true,
            cwd: 'assets/',
            src: 'fonts/**',
            dest: 'build/',
         },
         images: {
            expand: true,
            cwd: 'assets/',
            src: 'images/**',
            dest: 'build/',
         },
         headerImages: {
            expand: true,
            cwd: 'dependencies/header/assets/',
            src: 'images/**',
            dest: 'build/',
         },
         footerImages: {
            expand: true,
            cwd: 'dependencies/footer/assets/',
            src: 'images/**',
            dest: 'build/',
         },
         ref: {
            files: [
               {expand: true, src: ['**/*.+(js|css)'], dest: 'build/ref/bootstrap', cwd: 'bower_components/bootstrap-sass-official/assets'},
               {expand: true, src: ['*'], dest: 'build/ref/bootstrap/fonts', cwd: 'bower_components/bootstrap-sass-official/assets/fonts/bootstrap'},
               {expand: true, src: ['*.+(js|css|map)'], dest: 'build/ref/jquery', cwd: 'bower_components/jquery/dist'},
               {expand: true, src: ['./**/*.js'], dest: 'build/ref/momentjs/', cwd: 'bower_components/momentjs/min'},
            ]
         }
      },
      
      clean: {
         data: ['build/*']
      },

      sources: {
         jade: ['src/jade/**/*.jade'],
         js: ['src/js/*.js', '!src/js/ga.js', 'server/**/*.js'],
         scss: ['src/**/*.scss', '!src/scss/bourbon/**/*'],
         build: ['build/**/*.js'],
         gruntfile: ['Gruntfile.js']
      }
   });

   for (var dependency in grunt.config('pkg').dependencies) {
      if (dependency.indexOf('grunt-') === 0) { grunt.loadNpmTasks(dependency); }
   }
   
   if (process.env.NODE_ENV !== 'production') {
      for (var devDependency in grunt.config('pkg').devDependencies) {
         if (devDependency.indexOf('grunt-') === 0) { grunt.loadNpmTasks(devDependency); }
      }
   }

   grunt.registerTask('build', ['clean', 'sass', 'uglify', 'copy']);
   grunt.registerTask('test', ['scsslint', 'jshint']);
   grunt.registerTask('default', ['build']);
};
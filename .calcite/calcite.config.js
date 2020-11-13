const fs = require('fs');
const path = require('path');

const fromGoogleBenchFile = (filePath, builder, calciteContext) => {
    const data = calciteContext.readJson(filePath);
    
    const testSuiteName = path.basename(data.context.executable);
    
    data.benchmarks.forEach((bench) => {
        builder.addOrUpdateDataPoint(
            testSuiteName ,
            bench.run_name,
            'real_time', {
                values: [bench['real_time']],
                unit: bench['time_unit'],
            }
        );

        builder.addOrUpdateDataPoint(
            testSuiteName ,
            bench.run_name,
            'cpu_time', {
                values: [bench['cpu_time']],
                unit: bench['time_unit'],
            }
        );
    });
};

// TODO: use https://www.npmjs.com/package/glob ?
async function buildFilesList(pathsList, fileRegex)
{
    const pathsArray = pathsList.split(path.delimiter);
    return (await Promise.all( pathsArray.map( async pathEntry => {
        try{
            const filesInPath = await fs.promises.readdir(pathEntry, { withFileTypes: true });
            const test = filesInPath.map( file => 
            {
                if(file.isDirectory())
                {
                    // no recursion allowed
                }
                else if(file.isFile() && fileRegex.test(file.name))
                {
                    return path.join(pathEntry, file.name);
                }
                return null;
            }
            ).filter( x => x !== null );
            return test;

        } catch(_)
        {
            return [pathEntry];
        }
    }))).flat();
}

module.exports = async function (context) {
    if (!process.env.BENCH_OUTPUT_FILES) {
        console.error('[Error] - The BENCH_OUTPUT_FILES env variable was not set');
        process.exit(1);
    }

    const benchFiles = await buildFilesList(process.env.BENCH_OUTPUT_FILES, /.*\.json$/);
    
    const builder = context.helpers.TestSuitesBuilderFactory();
    benchFiles.forEach(filePath => fromGoogleBenchFile(filePath, builder, context.helpers));

    return {
        testSuites: builder.assembleAndGetAll()
    };
};

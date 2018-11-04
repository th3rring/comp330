import java.io.IOException;
import java.util.StringTokenizer;
import java.lang.NumberFormatException;
import java.util.NoSuchElementException;

import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.conf.Configured;
import java.util.ArrayList;
import java.util.List;
import java.text.SimpleDateFormat;
import java.text.ParseException;
import java.util.Map;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.lang.Comparable;


public class TaskThree extends Configured implements Tool {


	static int printUsage() {
		System.out.println("taskone [-m <maps>] [-r <reduces>] <input> <output>");
		ToolRunner.printGenericCommandUsage(System.out);
		return -1;
	}

	public static class TaskThreeMapper
			extends Mapper<Object, Text, Text, Text> {
			public class Tuple{
				public final String x;
				public final Double y;
				public Tuple(String x, Double y) {
					this.x = x;
					this.y = y;
				}

				public int compareTo(Tuple t)
				{
					return this.y.compareTo(t.y);
				}
			}

			private Comparator<Tuple> byRevenue = (Tuple t1, Tuple t2) -> t1.compareTo(t2);
			private PriorityQueue<Tuple> queue = new PriorityQueue(5, byRevenue);

			private final static Text key = new Text();
			private Text driverPair = new Text();

			public void map(Object key, Text value, Context context
				       ) throws IOException, InterruptedException {
				//Lines from the csv come in and the tokenizer breaks them up

				String[] input = value.toString().split("\\s");

				if(input.length == 2)
				{
					double curRevenue;
					try
					{
						curRevenue = Double.parseDouble(input[1]);
					}
					catch(NumberFormatException e)
					{
						return;
					}

					queue.add(new Tuple(input[0], curRevenue));

					if(queue.size() > 5)
						queue.remove();

				}



				       }
			public void cleanup(Context context) throws IOException, InterruptedException
			{
				key.set("key");
				while(queue.size() > 0)
				{

					Tuple cur = queue.poll();
					driverPair.set(cur.x.toString() + "=" + cur.y.toString());
					context.write(key, driverPair);
				}
			}
	}

	public static class TaskThreeReducer
			extends Reducer<Text,Text,Text,Text> {
			public class Tuple{
				public final String x;
				public final Double y;
				public Tuple(String x, Double y) {
					this.x = x;
					this.y = y;
				}

				public int compareTo(Tuple t)
				{
					return this.y.compareTo(t.y);
				}
			}

			private Comparator<Tuple> byRevenue = (Tuple t1, Tuple t2) -> t1.compareTo(t2);
			private PriorityQueue<Tuple> queue = new PriorityQueue(5, byRevenue);

			private Text revenue = new Text();
			private Text driver = new Text();

			public void reduce(Text key, Iterable<Text> values,
					Context context
					) throws IOException, InterruptedException {

				for (Text pair : values) {
					String[] pairArray = pair.toString().split("=");
						queue.add(new Tuple(pairArray[0], Double.parseDouble(pairArray[1])));

					if(queue.size() > 5)
						queue.remove();
				}

				while(queue.size() > 0)
				{

					Tuple cur = queue.remove();
					revenue.set(cur.y.toString());
					driver.set(cur.x);
					context.write(driver,revenue);
				}

					}
	}

	public int run(String[] args) throws Exception {

		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "Task Three");
		job.setJarByClass(TaskThree.class);
		job.setMapperClass(TaskThreeMapper.class);
		//job.setCombinerClass(TaskThreeReducer.class);
		job.setReducerClass(TaskThreeReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		List<String> other_args = new ArrayList<String>();
		for(int i=0; i < args.length; ++i) {
			try {
				if ("-r".equals(args[i])) {
					job.setNumReduceTasks(Integer.parseInt(args[++i]));
				} else {
					other_args.add(args[i]);
				}
			} catch (NumberFormatException except) {
				System.out.println("ERROR: Double expected instead of " + args[i]);
				return printUsage();
			} catch (ArrayIndexOutOfBoundsException except) {
				System.out.println("ERROR: Required parameter missing from " +
						args[i-1]);
				return printUsage();
			}
		}
		// Make sure there are exactly 2 parameters left.
		if (other_args.size() != 2) {
			System.out.println("ERROR: Wrong number of parameters: " +
					other_args.size() + " instead of 2.");
			return printUsage();
		}
		FileInputFormat.setInputPaths(job, other_args.get(0));
		FileOutputFormat.setOutputPath(job, new Path(other_args.get(1)));
		return (job.waitForCompletion(true) ? 0 : 1);
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new TaskThree(), args);
		System.exit(res);
	}

}

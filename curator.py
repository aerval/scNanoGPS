#! /usr/bin/env python3

import time, os, sys, subprocess, glob
import multiprocessing as mp
import pysam
from functools import partial
from contextlib import contextmanager
from scanner_core import misc
from curator_core import curator_io, preprocessing

@contextmanager
def poolcontext(*args, **kwargs):
	pool = mp.Pool(*args, **kwargs)
	yield pool
	pool.terminate()

if __name__ == "__main__":

	#===get params===
	parser = preprocessing.getOptions()
	options, arguments = parser.parse_args()

	#===pre-check===
	parser, options = preprocessing.precheck(parser, options)
	logger = open(options.log_f_name, "at")
	logger.write(" ".join(sys.argv) + "\n\n")

	#=== set env variables
	os.environ["OMP_NUM_THREADS"]        = str(options.ncores)
	os.environ["OPENBLAS_NUM_THREADS"]   = str(options.ncores)
	os.environ["MKL_NUM_THREADS"]        = str(options.ncores)
	os.environ["VECLIB_MAXIMUM_THREADS"] = str(options.ncores)
	os.environ["NUMEXPR_NUM_THREADS"]    = str(options.ncores)

	import numpy as np

	#===load CB_counting===
	start_time = time.time()
	print("\nTime stamp: " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "\n", flush = True)
	print("Loading CB_counting: " + options.CB_count + " ...", end = "", flush = True)
	CB_counting_df = curator_io.load_df(options.CB_count)
	CB_counting_df.set_index('idx', inplace=True)
	print("Done", flush = True)

	#===load CB_list===
	print("\nTime stamp: " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "\n", flush = True)
	print("Loading CB_list: " + options.CB_list + " ...", end = "", flush = True)
	CB_dict = dict()
	CB_list = curator_io.open_file(options.CB_list, "rt")
	#---skip def line---
	CB_list.readline()
	for line in CB_list:
		lines = line.rstrip().split(": ")

		rep_BC = CB_counting_df.loc[int(lines[0]), 'BC']

		all_id = curator_io.extract_id_list(lines[1])

		BC_seq_list = CB_counting_df.loc[all_id, 'BC']
		for ele in BC_seq_list.values:
			CB_dict[ele] = rep_BC

	CB_list.close()
	print("Done", flush = True)
	del CB_counting_df
	print("Time stamp: " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "\n", flush = True)
	hours, minutes, seconds = misc.get_time_elapse(start_time)
	misc.report_time_elapse(hours, minutes, seconds)

	#===parse reads===
	time_separation = time.time()
	print("Separation of reads by cell barcodes ...", flush = True)
	i = 0
	BC_list = curator_io.open_file(options.BC_list, "rt")
	# 0   1           2        3  4   5
	# rid orientation BC_start BC UMI mean_BC_quality
	#---skip def line---
	BC_list.readline()

	fastq_f = curator_io.open_file(options.fq_name, "rt")
	while True:
		i += 1
		print(i, end = "\r", flush = True)

		BC_line = BC_list.readline()
		if not BC_line:
			break

		BC_lines = BC_line.rstrip().split("\t")

		fastq_f.readline()
		seq_line = fastq_f.readline()
		fastq_f.readline()
		qua_line = fastq_f.readline()

		if BC_lines[3] in CB_dict:
			o_name = os.path.join(options.tmp_dir, CB_dict[BC_lines[3]] + ".fastq.gz")
			writer = curator_io.open_file(o_name, "at")
			writer.write("@" + BC_lines[0] + "_" + BC_lines[4] + "\n")
			writer.write(seq_line)
			writer.write("+" + BC_lines[0] + "_" + BC_lines[4] + "\n")
			writer.write(qua_line)
			writer.close()
	BC_list.close()
	fastq_f.close()
	print("            \rDone\n", flush = True)
	print("\nTime stamp: " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "\n", flush = True)
	hours, minutes, seconds = misc.get_time_elapse(start_time)
	misc.report_time_elapse(hours, minutes, seconds)
	hours, minutes, seconds = misc.get_time_elapse(time_separation)
	logger.write("Separation of reads by cell barcode spent %d : %d : %.2f\n\n" % (hours, minutes, seconds))

	#===batch mapping===
	logger.write("Summary table of curation per cell barcode:\n")
	logger.write("Cellbarcode\tRaw records\tLow softclipping records\tHigh softclipping records\tNon-duplicated records\tDuplicated records\tCurated records\tTime spent\n")
	print("Mapping reads for each individual cell barcode ...", flush = True)
	i = 0

	#=== get fastq files list ===
	fq_list = glob.glob(os.path.join(options.tmp_dir, "*.fastq.gz"))
	for fq in fq_list:
		i += 1
		time_curation = time.time()

		fq_pref = fq.split(".fastq.gz")[0].split(options.tmp_dir)[1].split('/')[1]
		stat_msg = str(i) + " of " + str(len(fq_list)) + " " + fq_pref + " ... "

		#---mapping---
		print(stat_msg + "mapping ..." + " " * 20, end = "\r", flush = True)
		cmd = options.minimap2 + " -ax splice " + \
		      "-t " + str(options.ncores) + " " + \
		      curator_io.check_mapping_db(options) + " " + fq
		code_msg, out_msg, err_msg = curator_io.sys_run(cmd)

		with open(os.path.join(options.tmp_dir, fq_pref) + ".sam", "wt") as SAM:
			SAM.write(out_msg.decode("utf-8"))

		with open(os.path.join(options.tmp_dir, fq_pref) + ".minimap2.log.txt", "wt") as MINIMAP2:
			MINIMAP2.write(err_msg.decode("utf-8"))

		#---samtools conversion---
		print(stat_msg + "convert sam to bam ..." + " " * 20, end = "\r", flush = True)
		   #--- with header ---
		cmd = options.samtools + " view -Sb " + \
		      os.path.join(options.tmp_dir, fq_pref) + ".sam " + \
		      "-@ " + str(options.ncores) + " " + \
		      "-T " + options.ref_genome + " " + \
		      "-o " + os.path.join(options.tmp_dir, fq_pref) + ".unsorted.bam"
		code_msg, out_msg, err_msg = curator_io.sys_run(cmd)

		if not options.keep_meta:
			cmd = "rm " + os.path.join(options.tmp_dir, fq_pref) + ".sam"
			os.system(cmd)

		print(stat_msg + "sorting bam ..." + " " * 20, end = "\r", flush = True)
		cmd = options.samtools + " sort " + \
		      os.path.join(options.tmp_dir, fq_pref) + ".unsorted.bam " + \
		      "-@ " + str(options.ncores) + " " + \
		      "-o " + os.path.join(options.tmp_dir, fq_pref) + ".minimap2.bam"
		code_msg, out_msg, err_msg = curator_io.sys_run(cmd)

		if not options.keep_meta:
			cmd = "rm " + os.path.join(options.tmp_dir, fq_pref) + ".unsorted.bam"
			os.system(cmd)

		print(stat_msg + "indexing ..." + " " * 20, end = "\r", flush = True)
		cmd = options.samtools + " index " + \
		      os.path.join(options.tmp_dir, fq_pref) + ".minimap2.bam"
		code_msg, out_msg, err_msg = curator_io.sys_run(cmd)

		#===retrieve read===
		seq_dict = curator_io.build_read_seq_dict(os.path.join(options.tmp_dir, fq_pref) + ".minimap2.bam")

		#===filter in/out regions, ex: rRNA/tRNA===
		curator_io.filter_bam_inc(fq_pref, options)
		curator_io.filter_bam_exc(fq_pref, options)

		#===filter softclipping===
		print(stat_msg + "filtering ..." + " " * 20, end = "\r", flush = True)
		curator_io.filter_softclipping(options.softclipping_thr, \
			os.path.join(options.tmp_dir, fq_pref) + ".minimap2.bam", \
			os.path.join(options.tmp_dir, fq_pref) + ".filtered.bam", \
			os.path.join(options.tmp_dir, fq_pref) + ".high_softclipping.bam")

		cmd = options.samtools + " view " + \
		      os.path.join(options.tmp_dir, fq_pref) + ".minimap2.bam | " + \
		      "wc -l"
		no_raw = subprocess.check_output(cmd, shell = True).decode("utf-8").strip()

		cmd = options.samtools + " view " + \
		      os.path.join(options.tmp_dir, fq_pref) + ".filtered.bam | " + \
		      "wc -l"
		no_filt = subprocess.check_output(cmd, shell = True).decode("utf-8").strip()

		cmd = options.samtools + " view " + \
		      os.path.join(options.tmp_dir, fq_pref) + ".high_softclipping.bam | " + \
		      "wc -l"
		no_h_soft = subprocess.check_output(cmd, shell = True).decode("utf-8").strip()

		cmd = options.samtools + " index " + \
		      os.path.join(options.tmp_dir, fq_pref) + ".filtered.bam"
		code_msg, out_msg, err_msg = curator_io.sys_run(cmd)

		if not options.skip_curation:
			#===UMI collapse===
			print(stat_msg + "collapsing UMI..." + " " * 20, end = "\r", flush = True)
			bam_i = pysam.AlignmentFile(os.path.join(options.tmp_dir, fq_pref) + ".filtered.bam",  "rb")
			compiled_list = curator_io.split_reads_by_pos(bam_i, seq_dict)
			bam_i.close()

#parallel_computing
			with poolcontext(processes = options.ncores) as pool:
				res = pool.map(partial(curator_io.collapse_UMI, options = options), compiled_list)

			#===output===
			uniq_con_res   = dict()
			uniq_con_reads = dict()
			bam_res_list = []
			for con_res, con_reads, bam_res in res:
				for key in con_res:
					uniq_con_res[key] = con_res[key]
				for key in con_reads:
					uniq_con_reads[key] = 1

				bam_res_list.extend(bam_res)

			con_o = open(os.path.join(options.tmp_dir, fq_pref) + ".consensus.fasta", "wt")
			for readID in uniq_con_res:
				con_o.write(">" + readID + "\n")
				con_o.write(uniq_con_res[readID])
			con_o.close()

			bam_i = pysam.AlignmentFile(os.path.join(options.tmp_dir, fq_pref) + ".filtered.bam",  "rb")
			bam_o = pysam.AlignmentFile(os.path.join(options.tmp_dir, fq_pref) + ".singleton.bam", "wb", template = bam_i)
			for read in bam_i.fetch():
				RID_pos_str = read.query_name     + ";" + \
				              read.reference_name + ";" + \
				          str(read.reference_start)
				if RID_pos_str in uniq_con_reads:
					continue
				read_str = read.query_name     + ";" + \
				           read.reference_name + ";" + \
				           str(read.reference_start)
				if read_str in bam_res_list:
					bam_o.write(read)
			bam_i.close()
			bam_o.close()

			if not options.keep_meta:
				cmd = "rm " + \
				      os.path.join(options.tmp_dir, fq_pref) + ".minimap2.bam"
				os.system(cmd)
				cmd = "rm " + \
				      os.path.join(options.tmp_dir, fq_pref) + ".minimap2.bam.bai"
				os.system(cmd)

			cmd = "grep '>' " + \
			      os.path.join(options.tmp_dir, fq_pref) + ".consensus.fasta | " \
			      "wc -l "
			no_dup = subprocess.check_output(cmd, shell = True).decode("utf-8").strip()

			cmd = options.samtools + " view " + \
			      os.path.join(options.tmp_dir, fq_pref) + ".singleton.bam | " + \
			      "wc -l"
			no_singleton = subprocess.check_output(cmd, shell = True).decode("utf-8").strip()

			if not options.keep_meta:
				cmd = "rm " + \
				      os.path.join(options.tmp_dir, fq_pref) + ".filtered.bam"
				os.system(cmd)
				cmd = "rm " + \
				      os.path.join(options.tmp_dir, fq_pref) + ".filtered.bam.bai"
				os.system(cmd)

			#---mapping consensus reads---
			print(stat_msg + "mapping consensus reads ..." + " " * 20, end = "\r", flush = True)
			cmd = options.minimap2 + " -ax splice " + \
			      "-t " + str(options.ncores) + " " + \
			      curator_io.check_mapping_db(options) + " " + \
			      os.path.join(options.tmp_dir, fq_pref) + ".consensus.fasta"
			code_msg, out_msg, err_msg = curator_io.sys_run(cmd)

			with open(os.path.join(options.tmp_dir, fq_pref) + ".consensus.sam", "wt") as SAM:
				SAM.write(out_msg.decode("utf-8"))

			with open(os.path.join(options.tmp_dir, fq_pref) + ".consensus.minimap2.log.txt", "wt") as MINIMAP2:
				MINIMAP2.write(err_msg.decode("utf-8"))

			if not options.keep_meta:
				cmd = "rm " + \
				      os.path.join(options.tmp_dir, fq_pref) + ".consensus.fasta"
				os.system(cmd)

			#---merge collaped reads---
			print(stat_msg + "merge back collapsed reads ..." + " " * 20, end = "\r", flush = True)
				#---with header---
			cmd = "cat " + \
			      os.path.join(options.tmp_dir, fq_pref) + ".consensus.sam >> " + \
			      os.path.join(options.tmp_dir, fq_pref) + ".curated.sam"
			os.system(cmd)

#			cmd = options.samtools + " view -Sb " + \
#			      os.path.join(options.tmp_dir, fq_pref) + ".consensus.sam " + \
#			      "-@ " + str(options.ncores) + " " + \
#			      "-T " + options.ref_genome + " " + \
#			      "-o " + os.path.join(options.tmp_dir, fq_pref) + ".consensus.bam"

			if not options.keep_meta:
				cmd = "rm " + \
				      os.path.join(options.tmp_dir, fq_pref) + ".consensus.?am"
				os.system(cmd)

				#---without header---
			cmd = options.samtools + " view " + \
			      os.path.join(options.tmp_dir, fq_pref) + ".singleton.bam " + \
			      "-o " + os.path.join(options.tmp_dir, fq_pref) + ".singleton.sam"
			os.system(cmd)
			cmd = "cat " + \
			      os.path.join(options.tmp_dir, fq_pref) + ".singleton.sam >> " + \
			      os.path.join(options.tmp_dir, fq_pref) + ".curated.sam"
			os.system(cmd)

			if not options.keep_meta:
				cmd = "rm " + \
				      os.path.join(options.tmp_dir, fq_pref) + ".singleton.?am"
				os.system(cmd)

			print(stat_msg + "convert curated sam to bam ..." + " " * 20, end = "\r", flush = True)
			   #--- with partial header ---
			cmd = options.samtools + " view -Sb " + \
			      os.path.join(options.tmp_dir, fq_pref) + ".curated.sam " + \
			      "-@ " + str(options.ncores) + " " + \
			      "-T " + options.ref_genome + " " + \
			      "-o " + os.path.join(options.tmp_dir, fq_pref) + ".curated.unsorted.bam"
			code_msg, out_msg, err_msg = curator_io.sys_run(cmd)

			if not options.keep_meta:
				cmd = "rm " + \
				      os.path.join(options.tmp_dir, fq_pref) + ".curated.sam"
				os.system(cmd)

			print(stat_msg + "sorting curated bam ..." + " " * 20, end = "\r", flush = True)
			cmd = options.samtools + " sort " + \
			      os.path.join(options.tmp_dir, fq_pref) + ".curated.unsorted.bam " + \
			      "-@ " + str(options.ncores) + " " + \
			      "-o " + os.path.join(options.tmp_dir, fq_pref) + ".curated.minimap2.bam"
			code_msg, out_msg, err_msg = curator_io.sys_run(cmd)

			if not options.keep_meta:
				cmd = "rm " + os.path.join(options.tmp_dir, fq_pref) + ".curated.unsorted.bam"
				os.system(cmd)

		else:
			no_singleton, no_dup = "NA", "NA"
			print("\t!!! Warning !!!\tSkip curation!!!")
			cmd = "mv " + os.path.join(options.tmp_dir, fq_pref) + ".filtered.bam " + os.path.join(options.tmp_dir, fq_pref) + ".curated.minimap2.bam"
			os.system(cmd)

		print(stat_msg + "indexing curated bam ..." + " " * 20, end = "\r", flush = True)
		cmd = options.samtools + " index " + \
		      os.path.join(options.tmp_dir, fq_pref) + ".curated.minimap2.bam"
		code_msg, out_msg, err_msg = curator_io.sys_run(cmd)

		cmd = options.samtools + " view " + \
		      os.path.join(options.tmp_dir, fq_pref) + ".curated.minimap2.bam | " + \
		      "wc -l"
		no_curated = subprocess.check_output(cmd, shell = True).decode("utf-8").strip()

		hours, minutes, seconds = misc.get_time_elapse(time_curation)
		time_spent = "%d : %d : %.2f" % (hours, minutes, seconds)

		logger.write("\t".join([fq_pref, no_raw, no_filt, no_h_soft, no_singleton, no_dup, no_curated, time_spent]) + "\n")

	print(str(len(fq_list)) + " of " + str(len(fq_list)) + " Done" + " " * 50, flush = True)
	print("\nTime stamp: " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), "\n", flush = True)
	hours, minutes, seconds = misc.get_time_elapse(start_time)
	misc.report_time_elapse(hours, minutes, seconds)

	logger.write("\nCuration process time spent: %d : %d : %.2f\n" % (hours, minutes, seconds))
	logger.close()

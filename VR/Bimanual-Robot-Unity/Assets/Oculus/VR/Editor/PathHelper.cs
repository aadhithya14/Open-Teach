/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;
using System.IO;

public static class PathHelper
{
	public static string MakeRelativePath(string fromPath, string toPath)
	{
		var fromUri = new Uri(Path.GetFullPath(fromPath));
		var toUri = new Uri(Path.GetFullPath(toPath));

		if (fromUri.Scheme != toUri.Scheme)
		{
			return toPath;
		}

		var relativeUri = fromUri.MakeRelativeUri(toUri);
		var relativePath = Uri.UnescapeDataString(relativeUri.ToString());

		if (toUri.Scheme.Equals("file", StringComparison.InvariantCultureIgnoreCase))
		{
			relativePath = relativePath.Replace(Path.AltDirectorySeparatorChar, Path.DirectorySeparatorChar);
		}

		return relativePath;
	}
}

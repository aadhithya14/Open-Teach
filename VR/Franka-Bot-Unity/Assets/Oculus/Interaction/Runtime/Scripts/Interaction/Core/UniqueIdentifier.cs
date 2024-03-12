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
using System.Collections.Generic;

namespace Oculus.Interaction
{
    public class UniqueIdentifier
    {
        public int ID { get; private set; }


        private UniqueIdentifier(int identifier)
        {
            ID = identifier;
        }

        private static System.Random Random = new System.Random();
        private static HashSet<int> _identifierSet = new HashSet<int>();

        public static UniqueIdentifier Generate()
        {
            while (true)
            {
                int identifier = Random.Next(Int32.MaxValue);
                if (_identifierSet.Contains(identifier)) continue;
                _identifierSet.Add(identifier);
                return new UniqueIdentifier(identifier);
            }
        }

        public static void Release(UniqueIdentifier identifier)
        {
            _identifierSet.Remove(identifier.ID);
        }
    }
}
